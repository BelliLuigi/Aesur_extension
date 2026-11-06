import argparse
import os
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
import pytorch_optimizer as opt_add

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False
torch.set_default_dtype(torch.float64)

from models import Autoencoder_CNN_NOBN
from utils import (
    train_one_epoch,
    validate,
    plot_training_curves,
    plot_mismatch_distribution,
    TasosOverlapTorch,
    EarlyStopping,
)


def compute_mismatches_logfreq(model_amplitude, model_logfreq, loader, device=None):
    """
    Evaluates Amplitude and Log-Frequency Autoencoders in 64-bit, reconstructs physical
    polar waveforms by integrating frequency (phi = int exp(log_omega) dt), and calculates
    individual waveform mismatches M = 1 - Overlap.
    """
    if device is None:
        device = next(model_amplitude.parameters()).device
    model_amplitude.eval()
    model_logfreq.eval()
    dtype = next(model_amplitude.parameters()).dtype

    mismatches = []
    with torch.no_grad():
        for b_ampl, b_logfreq in loader:
            b_ampl = b_ampl.to(device=device, dtype=dtype)
            b_logfreq = b_logfreq.to(device=device, dtype=dtype)
            if b_ampl.ndim == 2:
                b_ampl = b_ampl.unsqueeze(1)
            if b_logfreq.ndim == 2:
                b_logfreq = b_logfreq.unsqueeze(1)

            pred_ampl = model_amplitude(b_ampl)
            pred_logfreq = model_logfreq(b_logfreq)

            # Reconstruct frequency: omega = exp(log_omega)
            true_freq = torch.exp(b_logfreq.squeeze(1))
            pred_freq = torch.exp(pred_logfreq.squeeze(1))

            # Cumulative trapezoidal integration to reconstruct zeroed phase: phi(0) = 0
            zeros_true = torch.zeros((*true_freq.shape[:-1], 1), dtype=dtype, device=device)
            zeros_pred = torch.zeros((*pred_freq.shape[:-1], 1), dtype=dtype, device=device)

            true_phase = torch.cat([zeros_true, torch.cumulative_trapezoid(true_freq, dim=-1)], dim=-1)
            pred_phase = torch.cat([zeros_pred, torch.cumulative_trapezoid(pred_freq, dim=-1)], dim=-1)

            true_wf = torch.polar(b_ampl.squeeze(1), -true_phase)
            pred_wf = torch.polar(torch.clamp(pred_ampl.squeeze(1), min=0.0), -pred_phase)

            match = TasosOverlapTorch(true_wf, pred_wf)
            mismatch = torch.clamp(1.0 - match, min=0.0)
            mismatches.extend(mismatch.cpu().numpy().tolist())

    return np.array(mismatches, dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train 2 Autoencoders for Amplitude and Log Frequency in 64-bit (NRSur7dq4)")
parser.add_argument("--title", type=str, default=None, help="Title / prefix for this run (used in output filenames)")
parser.add_argument("--model", type=str, choices=["nobn"], default="nobn", help="Architecture: 'autoencoder_cnn_nobn'/'nobn' (No BatchNorm) (default: nobn)")
parser.add_argument("--opt", type=str, choices=["soap"], default="soap", help="Optimizer choice: 'soap' (default: soap)")
parser.add_argument("--scheduler", type=str, choices=["plateau"], default="plateau", help="LR scheduler choice: 'plateau' (default: plateau)")
parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs (default: 5000)")
parser.add_argument("--early_stop", type=int, default=400, help="Early stopping patience in epochs (default: 400, set 0 to disable)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Enable torch.compile for model acceleration (default: True, use --no-compile to disable)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"AE64_logfreq_{args.opt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_name = "Autoencoder_CNN_NOBN"
print(f"=== Starting Run: {title} | Model: {model_name} (float64) | Targets: Amplitude & Log-Frequency | Optimizer: {args.opt.upper()} | Scheduler: {args.scheduler.upper()} | Epochs: {args.epochs} ===")

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
# Load Dataset from HDF5 (64-bit precision)
# ---------------------------------------------------------------------------
filepath = "datasets/NRSur7dq4v2_dataset.hdf"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    n_samples, sample_len = waveforms_ds.shape
    amplitude = np.empty((n_samples, sample_len), dtype=np.float64)
    log_freq = np.empty((n_samples, sample_len), dtype=np.float64)
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        wf_chunk = waveforms_ds[start:end]
        amplitude[start:end] = np.abs(wf_chunk).astype(np.float64)
        # Instantaneous frequency: omega(t) = d phi / dt
        phase_raw = -np.unwrap(np.angle(wf_chunk), axis=-1)
        freq = np.gradient(phase_raw, axis=-1)
        np.maximum(freq, 1e-4, out=freq)
        log_freq[start:end] = np.log(freq).astype(np.float64)

ampl_tensor = torch.from_numpy(amplitude)
logfreq_tensor = torch.from_numpy(log_freq)

# Split train/val/test once
joint_dataset = TensorDataset(ampl_tensor, logfreq_tensor)
j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

# Derive single-branch subsets using the exact same indices
a_train  = Subset(ampl_tensor, j_train.indices)
lf_train = Subset(logfreq_tensor, j_train.indices)
a_val    = Subset(ampl_tensor, j_val.indices)
lf_val   = Subset(logfreq_tensor, j_val.indices)

test_loader = DataLoader(j_test, batch_size=1024, shuffle=False)
a_train_loader = DataLoader(a_train, batch_size=64, shuffle=True)
a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False)
lf_train_loader = DataLoader(lf_train, batch_size=64, shuffle=True)
lf_val_loader = DataLoader(lf_val, batch_size=2048, shuffle=False)


# ---------------------------------------------------------------------------
# Models, Loss Functions & Regularization (64-bit)
# ---------------------------------------------------------------------------

raw_model_amplitude = Autoencoder_CNN_NOBN().to(device).double()
raw_model_logfreq = Autoencoder_CNN_NOBN().to(device).double()

if args.compile:
    print("Compiling 64-bit models with torch.compile...")
    model_amplitude = torch.compile(raw_model_amplitude)
    model_logfreq = torch.compile(raw_model_logfreq)
else:
    model_amplitude = raw_model_amplitude
    model_logfreq = raw_model_logfreq

# Dedicated per-branch losses
loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3)
loss_fn_logfreq = nn.SmoothL1Loss(beta=1e-3)

lambda_latent = 1e-5

# ---------------------------------------------------------------------------
# Optimizers & Schedulers
# ---------------------------------------------------------------------------
if args.opt.lower() == "soap":
    opt_fn_amplitude = opt_add.SOAP(
        model_amplitude.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=1e-9, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
    opt_fn_logfreq = opt_add.SOAP(
        model_logfreq.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=1e-9, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
else:
    opt_fn_amplitude = AdamW(model_amplitude.parameters(), lr=4e-3, weight_decay=0)
    opt_fn_logfreq = AdamW(model_logfreq.parameters(), lr=4e-3, weight_decay=0)


scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
    opt_fn_amplitude, factor=0.5, patience=150, threshold=0.0001, threshold_mode="rel", min_lr=5e-7, cooldown=50
)
scheduler_logfreq = lr_scheduler.ReduceLROnPlateau(
    opt_fn_logfreq, factor=0.5, patience=150, threshold=0.0001, threshold_mode="rel", min_lr=5e-7, cooldown=50
)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
best_vloss_amplitude = float("inf")
best_vloss_logfreq = float("inf")

history = {
    "epoch": [],
    "train_loss_amplitude": [],
    "val_loss_amplitude": [],
    "lr_amplitude": [],
    "train_loss_logfreq": [],
    "val_loss_logfreq": [],
    "lr_logfreq": [],
}

# Early Stopping Initialization
if args.early_stop > 0:
    min_lr_target = 5e-7 if args.opt.lower() == "soap" else 1e-6
    early_stop_amplitude = EarlyStopping(
        patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=150
    )
    early_stop_logfreq = EarlyStopping(
        patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=150
    )
    print(f"Early stopping enabled: global_patience={args.early_stop}, floor_patience=150 at min_lr={min_lr_target:.1e} (rel threshold=0.01%)")
else:
    print("Early stopping disabled.")

print(f"Starting training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    avg_loss_ampl = train_one_epoch(
        model_amplitude, opt_fn_amplitude, loss_fn_amplitude, device, epoch, None, a_train_loader, lambda_latent=lambda_latent
    )
    avg_loss_lf = train_one_epoch(
        model_logfreq, opt_fn_logfreq, loss_fn_logfreq, device, epoch, None, lf_train_loader, lambda_latent=lambda_latent
    )

    # Validate
    avg_vloss_ampl = validate(model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=lambda_latent)
    avg_vloss_lf = validate(model_logfreq, lf_val_loader, loss_fn_logfreq, device=device, lambda_latent=lambda_latent)

    # Step LR Scheduler
    if args.scheduler == "plateau":
        scheduler_amplitude.step(avg_vloss_ampl)
        scheduler_logfreq.step(avg_vloss_lf)
    else:
        scheduler_amplitude.step()
        scheduler_logfreq.step()

    current_lr_ampl = scheduler_amplitude.get_last_lr()[0]
    current_lr_lf = scheduler_logfreq.get_last_lr()[0]

    # Record history
    history["epoch"].append(epoch + 1)
    history["train_loss_amplitude"].append(avg_loss_ampl)
    history["val_loss_amplitude"].append(avg_vloss_ampl)
    history["lr_amplitude"].append(current_lr_ampl)
    history["train_loss_logfreq"].append(avg_loss_lf)
    history["val_loss_logfreq"].append(avg_vloss_lf)
    history["lr_logfreq"].append(current_lr_lf)

    # Save best Amplitude model & optimizer state
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

    # Save best Log-Frequency model & optimizer state
    if avg_vloss_lf < best_vloss_logfreq:
        best_vloss_logfreq = avg_vloss_lf
        torch.save(raw_model_logfreq.state_dict(), f"{title}_logfreq_best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model_logfreq.state_dict(),
                "optimizer_state_dict": opt_fn_logfreq.state_dict(),
                "scheduler_state_dict": scheduler_logfreq.state_dict(),
                "best_vloss": best_vloss_logfreq,
            },
            f"{title}_logfreq_checkpoint.pt",
        )

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"  Amplitude   | Train Loss: {avg_loss_ampl:.6f} | Val Loss: {avg_vloss_ampl:.6f} | Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}")
        print(f"  Log-Freq    | Train Loss: {avg_loss_lf:.6f} | Val Loss: {avg_vloss_lf:.6f} | Best: {best_vloss_logfreq:.6f} | LR: {current_lr_lf:.2e}")

    # Check Early Stopping
    if args.early_stop > 0:
        stop_a = early_stop_amplitude.step(avg_vloss_ampl, current_lr_ampl)
        stop_lf = early_stop_logfreq.step(avg_vloss_lf, current_lr_lf)
        if stop_a and stop_lf:
            print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
            print(f"  Amplitude: {early_stop_amplitude.stop_reason}")
            print(f"  Log-Freq:  {early_stop_logfreq.stop_reason}")
            break


# ---------------------------------------------------------------------------
# Save 1 Unified CSV History
# ---------------------------------------------------------------------------
csv_filename = f"{title}_metrics.csv"
df_history = pd.DataFrame(history)
df_history.to_csv(csv_filename, index=False)
print(f"\nUnified metrics CSV saved: {csv_filename}")

# ---------------------------------------------------------------------------
# Generate Professional Plots (Log Scale, Red/Blue/Black, Min Val Tagged)
# ---------------------------------------------------------------------------
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

plot_logfreq_filename = f"{title}_logfreq_plot.png"
plot_training_curves(
    epochs=history["epoch"],
    train_losses=history["train_loss_logfreq"],
    val_losses=history["val_loss_logfreq"],
    lrs=history["lr_logfreq"],
    run_title=title,
    metric_name="LogFrequency",
    output_path=plot_logfreq_filename,
)

print(f"\nTraining completed successfully for run '{title}'.")

# ---------------------------------------------------------------------------
# Evaluate Best Saved Models and Plot Mismatch Distribution
# ---------------------------------------------------------------------------
best_ampl_path = f"{title}_amplitude_best.pt"
best_logfreq_path = f"{title}_logfreq_best.pt"

print(f"\nLoading best saved models for mismatch evaluation...")
raw_model_amplitude.load_state_dict(torch.load(best_ampl_path, map_location=device))
raw_model_logfreq.load_state_dict(torch.load(best_logfreq_path, map_location=device))

print("Computing test set waveform mismatches (integrating predicted frequency to phase)...")
mismatches = compute_mismatches_logfreq(model_amplitude, model_logfreq, test_loader, device=device)

plot_mismatch_filename = f"{title}_mismatch_distribution.png"
plot_mismatch_distribution(
    mismatches=mismatches,
    run_title=title,
    output_path=plot_mismatch_filename,
)

print(f"\nSaved artifacts:")
print(f"  - Best Amplitude Model:         {title}_amplitude_best.pt")
print(f"  - Amplitude Full Checkpoint:    {title}_amplitude_checkpoint.pt (includes optimizer state)")
print(f"  - Best Log-Frequency Model:     {title}_logfreq_best.pt")
print(f"  - Log-Freq Full Checkpoint:     {title}_logfreq_checkpoint.pt (includes optimizer state)")
print(f"  - History CSV:                  {csv_filename}")
print(f"  - Amplitude Plot:               {plot_ampl_filename}")
print(f"  - Log-Frequency Plot:           {plot_logfreq_filename}")
print(f"  - Mismatch Plot:                {plot_mismatch_filename}")
print(f"  - Test Set Mismatch Stats:      Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
