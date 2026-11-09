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
    compute_mismatches,
    plot_mismatch_distribution,
    EarlyStopping,
    WeightedSmoothL1Loss,
)


# ---------------------------------------------------------------------------
# Residual Phase Autoencoder Module (Attempt 1: Mean Detrending)
# ---------------------------------------------------------------------------
class ResidualPhaseAutoencoder(nn.Module):
    """
    Phase Autoencoder with Mean-Trajectory Detrending (Residual Learning).
    Input:  phase (B, 1, 2048) or (B, 2048)
    Internal: delta_phase = phase - mean_phase (compresses target variance by >10x)
              out_residual = base_autoencoder(delta_phase)
    Output: reconstructed phase = out_residual + mean_phase
    """
    def __init__(self, base_model: nn.Module, mean_phase: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean_phase", mean_phase.view(1, 1, -1))

    def forward(self, phase, return_latent: bool = False):
        if phase.ndim == 2:
            phase = phase.unsqueeze(1)
        delta_phase = phase - self.mean_phase
        if return_latent:
            rec_res, z = self.base_model(delta_phase, return_latent=True)
            return rec_res + self.mean_phase, z
        rec_res = self.base_model(delta_phase)
        return rec_res + self.mean_phase


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train 2 Autoencoders for Amplitude and Residual Phase (SOAP + Weighted Loss)")
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
        title = f"7ae_residual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_name = "Autoencoder_CNN_NOBN (Phase Residual Detrending)"
print(f"=== Starting Run: {title} | Model: {model_name} | Optimizer: SOAP | Scheduler: Plateau | Epochs: {args.epochs} ===")

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
# Load Dataset from HDF5
# ---------------------------------------------------------------------------
filepath = "datasets/NRSur7dq4v2_dataset.hdf"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    params_ds = f["Parameters"]
    n_samples, sample_len = waveforms_ds.shape
    amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
    phase = np.empty((n_samples, sample_len), dtype=np.float32)
    params = params_ds[:]
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        amplitude[start:end] = np.abs(waveforms_ds[start:end]).astype(np.float32)
        phase[start:end] = get_phase(waveforms_ds[start:end]).astype(np.float32)

ampl_tensor = torch.from_numpy(amplitude)
phase_tensor = torch.from_numpy(phase)

# Hardcoded Primary In-Plane Spin Loss Weighting (Threshold 0.30, Exponent 1.5, Max 3.5, Normalized)
chi1_perp = np.sqrt(params[:, 1] ** 2 + params[:, 2] ** 2).astype(np.float32)
ramp = np.clip((chi1_perp - 0.30) / (0.80 - 0.30), 0.0, 1.0)
sample_weights = 1.0 + (3.5 - 1.0) * (ramp ** 1.5)
sample_weights = (sample_weights / sample_weights.mean()).astype(np.float32)
weights_tensor = torch.from_numpy(sample_weights)
print(f"Hardcoded sample weights applied (ramp from chi1_perp=0.30, power=1.5): min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

# Split train/val/test once
joint_dataset = TensorDataset(ampl_tensor, phase_tensor, weights_tensor)
j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

# Compute Training-Split Mean Phase Trajectory for Detrending
mean_phase_np = phase[j_train.indices].mean(axis=0).astype(np.float32)
mean_phase_tensor = torch.from_numpy(mean_phase_np).to(device)
np.save(f"{title}_mean_phase.npy", mean_phase_np)
print(f"Computed mean phase trajectory from {len(j_train.indices)} training waveforms (saved to {title}_mean_phase.npy).")

# Single-branch datasets with sample weights (using Subset to avoid duplicating memory)
a_dataset = TensorDataset(ampl_tensor, weights_tensor)
p_dataset = TensorDataset(phase_tensor, weights_tensor)

a_train = Subset(a_dataset, j_train.indices)
p_train = Subset(p_dataset, j_train.indices)
a_val   = Subset(a_dataset, j_val.indices)
p_val   = Subset(p_dataset, j_val.indices)

# Test loader yields (amplitude, phase) for standard physical mismatch evaluation
test_dataset = Subset(TensorDataset(ampl_tensor, phase_tensor), j_test.indices)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

a_train_loader = DataLoader(a_train, batch_size=128, shuffle=True)
a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False)
p_train_loader = DataLoader(p_train, batch_size=128, shuffle=True)
p_val_loader = DataLoader(p_val, batch_size=2048, shuffle=False)

# ---------------------------------------------------------------------------
# Models, Loss Functions & Regularization
# ---------------------------------------------------------------------------
raw_model_amplitude = Autoencoder_CNN_NOBN().to(device)
raw_model_phase_base = Autoencoder_CNN_NOBN().to(device)
raw_model_phase = ResidualPhaseAutoencoder(raw_model_phase_base, mean_phase_tensor).to(device)

if sys.version_info >= (3, 13):
    print("Notice: torch.compile/Dynamo is not supported on Python 3.13+. Continuing with uncompiled models.")
    model_amplitude = raw_model_amplitude
    model_phase = raw_model_phase
else:
    try:
        print("Compiling models with torch.compile...")
        model_amplitude = torch.compile(raw_model_amplitude)
        model_phase = torch.compile(raw_model_phase)
    except Exception as e:
        print(f"Notice: torch.compile failed ({e}). Continuing with uncompiled models.")
        model_amplitude = raw_model_amplitude
        model_phase = raw_model_phase

loss_fn_amplitude = WeightedSmoothL1Loss(beta=1e-3)
loss_fn_phase = WeightedSmoothL1Loss(beta=1e-3)
lambda_latent = 1e-6

# ---------------------------------------------------------------------------
# Optimizers (Hardcoded SOAP) & Schedulers (Hardcoded Plateau)
# ---------------------------------------------------------------------------
opt_fn_amplitude = opt_add.SOAP(
    raw_model_amplitude.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=25, merge_dims=False, eps=0.0001
)
opt_fn_phase = opt_add.SOAP(
    raw_model_phase_base.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=25, merge_dims=False, eps=0.0001
)

scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
    opt_fn_amplitude, factor=0.5, patience=50, threshold=0.005, threshold_mode="rel", min_lr=5e-7, cooldown=20
)
scheduler_phase = lr_scheduler.ReduceLROnPlateau(
    opt_fn_phase, factor=0.5, patience=50, threshold=0.005, threshold_mode="rel", min_lr=5e-7, cooldown=20
)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
best_vloss_amplitude = float("inf")
best_vloss_phase = float("inf")

history = {
    "epoch": [],
    "train_loss_amplitude": [],
    "val_loss_amplitude": [],
    "lr_amplitude": [],
    "train_loss_phase": [],
    "val_loss_phase": [],
    "lr_phase": [],
}

stopped_amplitude = False
stopped_phase = False

if args.early_stop > 0:
    min_lr_target = 5e-7
    early_stop_amplitude = EarlyStopping(
        patience=args.early_stop, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    early_stop_phase = EarlyStopping(
        patience=args.early_stop, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    print(f"Early stopping enabled: patience={args.early_stop}, floor_patience=50 at min_lr={min_lr_target:.1e}")
else:
    print("Early stopping disabled.")

print(f"Starting residual phase training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    avg_loss_ampl = train_one_epoch(
        model_amplitude, opt_fn_amplitude, loss_fn_amplitude, device, epoch, None, a_train_loader, lambda_latent=lambda_latent
    )
    avg_loss_ph = train_one_epoch(
        model_phase, opt_fn_phase, loss_fn_phase, device, epoch, None, p_train_loader, lambda_latent=lambda_latent
    )

    avg_vloss_ampl = validate(model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=lambda_latent)
    avg_vloss_ph = validate(model_phase, p_val_loader, loss_fn_phase, device=device, lambda_latent=lambda_latent)

    scheduler_amplitude.step(avg_vloss_ampl)
    scheduler_phase.step(avg_vloss_ph)

    current_lr_ampl = scheduler_amplitude.get_last_lr()[0]
    current_lr_ph = scheduler_phase.get_last_lr()[0]

    history["epoch"].append(epoch + 1)
    history["train_loss_amplitude"].append(avg_loss_ampl)
    history["val_loss_amplitude"].append(avg_vloss_ampl)
    history["lr_amplitude"].append(current_lr_ampl)
    history["train_loss_phase"].append(avg_loss_ph)
    history["val_loss_phase"].append(avg_vloss_ph)
    history["lr_phase"].append(current_lr_ph)

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

    if avg_vloss_ph < best_vloss_phase:
        best_vloss_phase = avg_vloss_ph
        torch.save(
            {
                "base_model_state_dict": raw_model_phase_base.state_dict(),
                "mean_phase": mean_phase_np,
            },
            f"{title}_phase_best.pt",
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "base_model_state_dict": raw_model_phase_base.state_dict(),
                "mean_phase": mean_phase_np,
                "optimizer_state_dict": opt_fn_phase.state_dict(),
                "scheduler_state_dict": scheduler_phase.state_dict(),
                "best_vloss": best_vloss_phase,
            },
            f"{title}_phase_checkpoint.pt",
        )

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"  Amplitude | Train Loss: {avg_loss_ampl:.6f} | Val Loss: {avg_vloss_ampl:.6f} | Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}")
        print(f"  Phase Res | Train Loss: {avg_loss_ph:.6f} | Val Loss: {avg_vloss_ph:.6f} | Best: {best_vloss_phase:.6f} | LR: {current_lr_ph:.2e}")

    if args.early_stop > 0:
        if not stopped_amplitude:
            stopped_amplitude = early_stop_amplitude.step(avg_vloss_ampl, current_lr_ampl)
        if not stopped_phase:
            stopped_phase = early_stop_phase.step(avg_vloss_ph, current_lr_ph)

        if stopped_amplitude and stopped_phase:
            print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
            print(f"  Amplitude: {early_stop_amplitude.stop_reason}")
            print(f"  Phase:     {early_stop_phase.stop_reason}")
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

plot_phase_filename = f"{title}_phase_plot.png"
plot_training_curves(
    epochs=history["epoch"],
    train_losses=history["train_loss_phase"],
    val_losses=history["val_loss_phase"],
    lrs=history["lr_phase"],
    run_title=title,
    metric_name="Phase_Residual",
    output_path=plot_phase_filename,
)

# ---------------------------------------------------------------------------
# Evaluate Best Saved Models and Plot Mismatch Distribution
# ---------------------------------------------------------------------------
best_ampl_path = f"{title}_amplitude_best.pt"
best_phase_path = f"{title}_phase_best.pt"

print(f"\nLoading best saved models for mismatch evaluation...")
raw_model_amplitude.load_state_dict(torch.load(best_ampl_path, map_location=device))
saved_phase_ckpt = torch.load(best_phase_path, map_location=device)
raw_model_phase_base.load_state_dict(saved_phase_ckpt["base_model_state_dict"])

print("Computing test set waveform mismatches...")
mismatches = compute_mismatches(model_amplitude, model_phase, test_loader, device=device)

plot_mismatch_filename = f"{title}_mismatch_distribution.png"
plot_mismatch_distribution(
    mismatches=mismatches,
    run_title=title,
    output_path=plot_mismatch_filename,
)

print(f"\nSaved artifacts:")
print(f"  - Best Amplitude Model:      {title}_amplitude_best.pt")
print(f"  - Best Phase Model:          {title}_phase_best.pt (includes base weights & mean_phase)")
print(f"  - Mean Phase Array:          {title}_mean_phase.npy")
print(f"  - Amplitude Plot:            {plot_ampl_filename}")
print(f"  - Phase Plot:                {plot_phase_filename}")
print(f"  - Mismatch Plot:             {plot_mismatch_filename}")
print(f"  - Test Set Mismatch Stats:   Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
