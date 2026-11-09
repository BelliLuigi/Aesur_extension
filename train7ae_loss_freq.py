import argparse
import os
import sys
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
import pytorch_optimizer as opt_add

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

from models import Autoencoder_CNN_NOBN
from utils import (
    get_phase,
    train_one_epoch,
    validate,
    plot_training_curves,
    plot_mismatch_distribution,
    EarlyStopping,
    WeightedSmoothL1Loss,
    TasosOverlapTorch,
)


def compute_mismatches_freq(model_amplitude, model_frequency, loader, device=None):
    """
    Evaluates Amplitude and Frequency Autoencoders, reconstructs physical polar
    waveforms by cumulative sum integration of frequency: phi(t) = cumsum(freq),
    and calculates individual waveform mismatches M = 1 - Overlap.
    """
    if device is None:
        device = next(model_amplitude.parameters()).device
    model_amplitude.eval()
    model_frequency.eval()
    dtype = next(model_amplitude.parameters()).dtype

    mismatches = []
    with torch.no_grad():
        for b_ampl, b_freq in loader:
            b_ampl = b_ampl.to(device=device, dtype=dtype)
            b_freq = b_freq.to(device=device, dtype=dtype)
            if b_ampl.ndim == 2:
                b_ampl = b_ampl.unsqueeze(1)
            if b_freq.ndim == 2:
                b_freq = b_freq.unsqueeze(1)

            pred_ampl = model_amplitude(b_ampl)
            pred_freq = model_frequency(b_freq)

            # Reconstruct physical phase by exact cumulative sum integration
            true_phase = torch.cumsum(b_freq.squeeze(1), dim=-1)
            pred_phase = torch.cumsum(pred_freq.squeeze(1), dim=-1)

            true_wf = torch.polar(b_ampl.squeeze(1), -true_phase)
            pred_wf = torch.polar(torch.clamp(pred_ampl.squeeze(1), min=0.0), -pred_phase)

            match = TasosOverlapTorch(true_wf, pred_wf)
            mismatch = torch.clamp(1.0 - match, min=0.0)
            mismatches.extend(mismatch.cpu().numpy().tolist())

    return np.array(mismatches, dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI Arguments (Minimal Execution Controls)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train 2 Autoencoders for Amplitude and Frequency (SOAP + Weighted Loss)")
parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
parser.add_argument("--epochs", type=int, default=5000, help="Training epochs (default: 5000)")
parser.add_argument("--early_stop", type=int, default=300, help="Early stopping patience in epochs (default: 300, 0 to disable)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"AE_freq_soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"=== Starting Run: {title} | Targets: Amplitude & Frequency | Optimizer: SOAP | Scheduler: PLATEAU | Epochs: {args.epochs} ===")

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(device)
    print(f"Allocated GPU: {prop.name} | Compute Capability: {prop.major}.{prop.minor} | VRAM: {prop.total_memory / (1024**3):.1f} GB")
else:
    print("Using CPU device.")

# ---------------------------------------------------------------------------
# Load Dataset from HDF5 & Extract Amplitude and Instantaneous Frequency
# ---------------------------------------------------------------------------
filepath = "datasets/NRSur7dq4v2_dataset.hdf"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    params_ds = f["Parameters"]
    n_samples, sample_len = waveforms_ds.shape
    amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
    frequency = np.empty((n_samples, sample_len), dtype=np.float32)
    params = params_ds[:]
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        wf_chunk = waveforms_ds[start:end]
        amplitude[start:end] = np.abs(wf_chunk).astype(np.float32)
        # Physical phase in radians (unwrapped and zero-aligned at t=0)
        ph_chunk = get_phase(wf_chunk).astype(np.float32) * 100.0
        # Discrete forward difference: exact inverse of cumsum
        frequency[start:end] = np.pad(np.diff(ph_chunk, axis=-1), ((0, 0), (1, 0))).astype(np.float32)

ampl_tensor = torch.from_numpy(amplitude)
freq_tensor = torch.from_numpy(frequency)

# Hardcoded Primary In-Plane Spin Loss Weighting (Threshold 0.30, Exponent 1.5, Max 3.5, Normalized)
chi1_perp = np.sqrt(params[:, 1] ** 2 + params[:, 2] ** 2).astype(np.float32)
ramp = np.clip((chi1_perp - 0.30) / (0.80 - 0.30), 0.0, 1.0)
sample_weights = 1.0 + (3.5 - 1.0) * (ramp ** 1.5)
sample_weights = (sample_weights / sample_weights.mean()).astype(np.float32)
weights_tensor = torch.from_numpy(sample_weights)
print(f"Hardcoded sample weights applied (ramp from chi1_perp=0.30, power=1.5): min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

# Split train/val/test once (80/10/10)
joint_dataset = TensorDataset(ampl_tensor, freq_tensor, weights_tensor)
j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

a_dataset = TensorDataset(ampl_tensor, weights_tensor)
f_dataset = TensorDataset(freq_tensor, weights_tensor)

a_train = Subset(a_dataset, j_train.indices)
f_train = Subset(f_dataset, j_train.indices)
a_val   = Subset(a_dataset, j_val.indices)
f_val   = Subset(f_dataset, j_val.indices)

# Test set keeps joint (ampl, freq) for compute_mismatches_freq
test_dataset = Subset(TensorDataset(ampl_tensor, freq_tensor), j_test.indices)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

a_train_loader = DataLoader(a_train, batch_size=128, shuffle=True)
a_val_loader   = DataLoader(a_val, batch_size=4096, shuffle=False)
f_train_loader = DataLoader(f_train, batch_size=128, shuffle=True)
f_val_loader   = DataLoader(f_val, batch_size=4096, shuffle=False)

# ---------------------------------------------------------------------------
# 2 Models: Amplitude Autoencoder & Frequency Autoencoder
# ---------------------------------------------------------------------------
raw_model_amplitude = Autoencoder_CNN_NOBN().to(device)
raw_model_frequency = Autoencoder_CNN_NOBN().to(device)

if sys.version_info >= (3, 13):
    print("Notice: torch.compile/Dynamo is not supported on Python 3.13+. Continuing with uncompiled models.")
    model_amplitude = raw_model_amplitude
    model_frequency = raw_model_frequency
else:
    try:
        print("Compiling models with torch.compile...")
        model_amplitude = torch.compile(raw_model_amplitude)
        model_frequency = torch.compile(raw_model_frequency)
    except Exception as e:
        print(f"Notice: torch.compile failed ({e}). Continuing with uncompiled models.")
        model_amplitude = raw_model_amplitude
        model_frequency = raw_model_frequency

loss_fn_amplitude = WeightedSmoothL1Loss(beta=1e-3)
loss_fn_frequency = WeightedSmoothL1Loss(beta=1e-3)
lambda_latent = 1e-6

# ---------------------------------------------------------------------------
# Optimizers (Hardcoded SOAP) & Schedulers (Hardcoded Plateau)
# ---------------------------------------------------------------------------
opt_fn_amplitude = opt_add.SOAP(
    raw_model_amplitude.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=25, merge_dims=False, eps=0.0001
)
opt_fn_frequency = opt_add.SOAP(
    raw_model_frequency.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=25, merge_dims=False, eps=0.0001
)

scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
    opt_fn_amplitude, factor=0.5, patience=50, threshold=0.005, threshold_mode="rel", min_lr=5e-7, cooldown=20
)
scheduler_frequency = lr_scheduler.ReduceLROnPlateau(
    opt_fn_frequency, factor=0.5, patience=50, threshold=0.005, threshold_mode="rel", min_lr=5e-7, cooldown=20
)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
best_vloss_amplitude = float("inf")
best_vloss_frequency = float("inf")

history = {
    "epoch": [],
    "train_loss_amplitude": [],
    "val_loss_amplitude": [],
    "lr_amplitude": [],
    "train_loss_frequency": [],
    "val_loss_frequency": [],
    "lr_frequency": [],
}

stopped_amplitude = False
stopped_frequency = False

if args.early_stop > 0:
    min_lr_target = 5e-7
    early_stop_amplitude = EarlyStopping(
        patience=args.early_stop, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    early_stop_frequency = EarlyStopping(
        patience=args.early_stop, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    print(f"Early stopping enabled: patience={args.early_stop}, floor_patience=50 at min_lr={min_lr_target:.1e}")
else:
    print("Early stopping disabled.")

print(f"Starting Amplitude & Frequency training for {args.epochs} epochs with SOAP...")

for epoch in range(args.epochs):
    avg_loss_ampl = train_one_epoch(
        model_amplitude, opt_fn_amplitude, loss_fn_amplitude, device, epoch, None, a_train_loader, lambda_latent=lambda_latent
    )
    avg_loss_freq = train_one_epoch(
        model_frequency, opt_fn_frequency, loss_fn_frequency, device, epoch, None, f_train_loader, lambda_latent=lambda_latent
    )

    avg_vloss_ampl = validate(model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=lambda_latent)
    avg_vloss_freq = validate(model_frequency, f_val_loader, loss_fn_frequency, device=device, lambda_latent=lambda_latent)

    scheduler_amplitude.step(avg_vloss_ampl)
    scheduler_frequency.step(avg_vloss_freq)

    current_lr_ampl = scheduler_amplitude.get_last_lr()[0]
    current_lr_freq = scheduler_frequency.get_last_lr()[0]

    history["epoch"].append(epoch + 1)
    history["train_loss_amplitude"].append(avg_loss_ampl)
    history["val_loss_amplitude"].append(avg_vloss_ampl)
    history["lr_amplitude"].append(current_lr_ampl)
    history["train_loss_frequency"].append(avg_loss_freq)
    history["val_loss_frequency"].append(avg_vloss_freq)
    history["lr_frequency"].append(current_lr_freq)

    # Save best models
    if avg_vloss_ampl < best_vloss_amplitude:
        best_vloss_amplitude = avg_vloss_ampl
        torch.save(raw_model_amplitude.state_dict(), f"{title}_amplitude_best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model_amplitude.state_dict(),
                "optimizer_state_dict": opt_fn_amplitude.state_dict(),
                "scheduler_state_dict": scheduler_amplitude.state_dict(),
                "best_vloss": best_vloss_amplitude,
            },
            f"{title}_amplitude_checkpoint.pt",
        )

    if avg_vloss_freq < best_vloss_frequency:
        best_vloss_frequency = avg_vloss_freq
        torch.save(raw_model_frequency.state_dict(), f"{title}_frequency_best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model_frequency.state_dict(),
                "optimizer_state_dict": opt_fn_frequency.state_dict(),
                "scheduler_state_dict": scheduler_frequency.state_dict(),
                "best_vloss": best_vloss_frequency,
            },
            f"{title}_frequency_checkpoint.pt",
        )

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"  Amplitude | Train: {avg_loss_ampl:.6f} | Val: {avg_vloss_ampl:.6f} | Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}")
        print(f"  Frequency | Train: {avg_loss_freq:.6f} | Val: {avg_vloss_freq:.6f} | Best: {best_vloss_frequency:.6f} | LR: {current_lr_freq:.2e}")

    if args.early_stop > 0:
        if not stopped_amplitude:
            stopped_amplitude = early_stop_amplitude.step(avg_vloss_ampl, current_lr_ampl)
        if not stopped_frequency:
            stopped_frequency = early_stop_frequency.step(avg_vloss_freq, current_lr_freq)

        if stopped_amplitude and stopped_frequency:
            print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
            print(f"  Amplitude: {early_stop_amplitude.stop_reason}")
            print(f"  Frequency: {early_stop_frequency.stop_reason}")
            break

# ---------------------------------------------------------------------------
# Save Metrics & Plots
# ---------------------------------------------------------------------------
csv_filename = f"{title}_metrics.csv"
try:
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_filename, index=False)
    print(f"\nUnified metrics CSV saved: {csv_filename}")
except Exception as e:
    print(f"\nWarning: Failed to save metrics CSV ({e}).")

plot_ampl_filename = f"{title}_amplitude_plot.png"
plot_training_curves(
    epochs=history["epoch"],
    train_losses=history["train_loss_amplitude"],
    val_losses=history["val_loss_amplitude"],
    lrs=history["lr_amplitude"],
    run_title=title,
    metric_name="Amplitude",
    output_path=plot_ampl_filename,
)

plot_freq_filename = f"{title}_frequency_plot.png"
plot_training_curves(
    epochs=history["epoch"],
    train_losses=history["train_loss_frequency"],
    val_losses=history["val_loss_frequency"],
    lrs=history["lr_frequency"],
    run_title=title,
    metric_name="Frequency",
    output_path=plot_freq_filename,
)

# ---------------------------------------------------------------------------
# Evaluate Best Saved Models and Plot Mismatch Distribution
# ---------------------------------------------------------------------------
best_ampl_path = f"{title}_amplitude_best.pt"
best_freq_path = f"{title}_frequency_best.pt"

print(f"\nLoading best saved models for mismatch evaluation...")
raw_model_amplitude.load_state_dict(torch.load(best_ampl_path, map_location=device))
raw_model_frequency.load_state_dict(torch.load(best_freq_path, map_location=device))

print("Computing test set waveform mismatches from reconstructed frequency...")
mismatches = compute_mismatches_freq(model_amplitude, model_frequency, test_loader, device=device)

plot_mismatch_filename = f"{title}_mismatch_distribution.png"
plot_mismatch_distribution(
    mismatches=mismatches,
    run_title=title,
    output_path=plot_mismatch_filename,
)

print(f"\nSaved artifacts:")
print(f"  - Best Amplitude Model:      {title}_amplitude_best.pt")
print(f"  - Best Frequency Model:      {title}_frequency_best.pt")
print(f"  - Amplitude Plot:            {plot_ampl_filename}")
print(f"  - Frequency Plot:            {plot_freq_filename}")
print(f"  - Mismatch Plot:             {plot_mismatch_filename}")
print(f"  - Test Set Mismatch Stats:   Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
