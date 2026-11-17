import argparse
import os
import sys
import time
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

from models import AutoencoderMono, init_kaiming_weights
from utils import (
    get_phase,
    validate,
    plot_training_curves,
    compute_mismatches,
    plot_mismatch_distribution,
    EarlyStopping,
)

# ---------------------------------------------------------------------------
# CLI Arguments (Minimal Execution Controls)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train 2 Autoencoders for Amplitude and Phase (SOAP, AutoencoderMono)")
parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
parser.add_argument("--epochs", type=int, default=800, help="Training epochs (default: 800)")
parser.add_argument("--small", action=argparse.BooleanOptionalAction, default=False, help="Use small dataset variant (NRSur7dq4v2_dataset_small.hdf) (default: False)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable torch.compile (default: False)")
parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable TF32 precision on CUDA (default: False)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Hardcoded Hyperparameters & Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
PRECONDITION_FREQ = 15
BETA = 0.3
CLIP_NORM = 1.0
LR = 1e-3
WARMUP_STEPS = 200
EARLY_STOP = 0

# Configure Precision (TF32 vs. standard FP32)
if args.tf32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    print("Precision: TF32 enabled for CUDA matmul and cuDNN.")
else:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    print("Precision: Standard FP32 (highest precision, TF32 disabled).")

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"AE_mono_soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_name = "AutoencoderMono"
print(f"=== Starting Run: {title} | Model: {model_name} | Optimizer: SOAP | Schedule: Warmup+Cosine | Epochs: {args.epochs} ===")

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
filepath = "datasets/NRSur7dq4v2_dataset_small.hdf" if args.small else "datasets/NRSur7dq4v2_dataset.hdf"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

print(f"Loading waveforms from '{filepath}' (small={args.small})...")
with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    params_ds = f["Parameters"]
    total_avail, sample_len = waveforms_ds.shape
    n_samples = total_avail
    amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
    phase = np.empty((n_samples, sample_len), dtype=np.float32)
    params = params_ds[:n_samples]
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        amplitude[start:end] = np.abs(waveforms_ds[start:end]).astype(np.float32)
        phase[start:end] = get_phase(waveforms_ds[start:end]).astype(np.float32)

print(f"Loaded {n_samples} waveforms (length: {sample_len}).")

# Split Dataset (80% train / 10% val / 10% test)
idx_all = np.arange(n_samples)
gen = torch.Generator().manual_seed(42)
j_train_idx, j_val_idx, j_test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)

train_indices = np.array(j_train_idx.indices)
val_indices = np.array(j_val_idx.indices)
test_indices = np.array(j_test_idx.indices)

print(f"Dataset split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

# Compute per-timestep mean and standard deviation strictly on the training set
print("Normalizing Amplitude and Phase using train-set per-timestep statistics...")
mu_ampl = np.zeros(sample_len, dtype=np.float64)
sq_ampl = np.zeros(sample_len, dtype=np.float64)
mu_phase = np.zeros(sample_len, dtype=np.float64)
sq_phase = np.zeros(sample_len, dtype=np.float64)

for s in range(0, len(train_indices), 5000):
    idx_blk = np.sort(train_indices[s:s + 5000])
    blk_a = amplitude[idx_blk].astype(np.float64)
    mu_ampl += blk_a.sum(axis=0)
    sq_ampl += (blk_a ** 2).sum(axis=0)

    blk_p = phase[idx_blk].astype(np.float64)
    mu_phase += blk_p.sum(axis=0)
    sq_phase += (blk_p ** 2).sum(axis=0)

mu_ampl /= len(train_indices)
sd_ampl = np.sqrt(np.maximum(sq_ampl / len(train_indices) - mu_ampl ** 2, 0.0)) + 1e-6

mu_phase /= len(train_indices)
sd_phase = np.sqrt(np.maximum(sq_phase / len(train_indices) - mu_phase ** 2, 0.0)) + 1e-6

# Normalize in-place across all samples
for s in range(0, n_samples, 5000):
    amplitude[s:s + 5000] = (amplitude[s:s + 5000] - mu_ampl) / sd_ampl
    phase[s:s + 5000] = (phase[s:s + 5000] - mu_phase) / sd_phase

# Save normalization constants
norm_npz_path = f"{title}_norm.npz"
np.savez(
    norm_npz_path,
    mu_ampl=mu_ampl,
    sd_ampl=sd_ampl,
    mu_phase=mu_phase,
    sd_phase=sd_phase,
)
print(f"Saved normalization constants: {norm_npz_path}")

ampl_tensor = torch.from_numpy(amplitude)
phase_tensor = torch.from_numpy(phase)

# Derive single-branch subsets using the exact same indices
a_train = Subset(ampl_tensor, train_indices)
p_train = Subset(phase_tensor, train_indices)
a_val   = Subset(ampl_tensor, val_indices)
p_val   = Subset(phase_tensor, val_indices)

test_joint = TensorDataset(ampl_tensor[test_indices], phase_tensor[test_indices])
test_loader = DataLoader(test_joint, batch_size=1024, shuffle=False)

a_train_loader = DataLoader(a_train, batch_size=BATCH_SIZE, shuffle=True)
a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False)
p_train_loader = DataLoader(p_train, batch_size=BATCH_SIZE, shuffle=True)
p_val_loader = DataLoader(p_val, batch_size=2048, shuffle=False)

# ---------------------------------------------------------------------------
# Models, Loss Functions & Regularization
# ---------------------------------------------------------------------------

raw_model_amplitude = AutoencoderMono().to(device)
raw_model_phase = AutoencoderMono().to(device)
init_kaiming_weights(raw_model_amplitude)
init_kaiming_weights(raw_model_phase)

if args.compile:
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
else:
    print("torch.compile disabled. Running uncompiled eager models.")
    model_amplitude = raw_model_amplitude
    model_phase = raw_model_phase

# Dedicated per-branch losses
loss_fn_amplitude = nn.SmoothL1Loss(beta=BETA)
loss_fn_phase = nn.SmoothL1Loss(beta=BETA)
lambda_latent = 0.0

# ---------------------------------------------------------------------------
# Training Helper (Gradient Clipping + Step-based Scheduler)
# ---------------------------------------------------------------------------
def train_one_epoch_mono(
    model, optimizer, scheduler, loss_fn, device, loader, clip_norm: float = CLIP_NORM, lambda_latent: float = 0.0
):
    model.train(True)
    dtype = next(model.parameters()).dtype
    total_loss = 0.0

    for batch in loader:
        inputs = batch.to(device=device, dtype=dtype)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        if lambda_latent > 0.0:
            outputs, z = model(inputs, return_latent=True)
            recon_loss = loss_fn(outputs, inputs)
            loss = recon_loss + lambda_latent * torch.mean(z ** 2)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

        loss.backward()

        # Gradient clipping before optimizer step
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

# ---------------------------------------------------------------------------
# Optimizers (SOAP, eps=1e-8) & Schedulers (Warmup + Cosine Decay)
# ---------------------------------------------------------------------------
opt_fn_amplitude = opt_add.SOAP(
    model_amplitude.parameters(), lr=LR, betas=(0.95, 0.95), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=PRECONDITION_FREQ, merge_dims=False, eps=1e-8
)
opt_fn_phase = opt_add.SOAP(
    model_phase.parameters(), lr=LR, betas=(0.95, 0.95), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=PRECONDITION_FREQ, merge_dims=False, eps=1e-8
)

# Step-based Warmup + Cosine Decay Schedules
total_steps = args.epochs * len(a_train_loader)
warmup_steps = min(WARMUP_STEPS, max(1, total_steps // 2))
decay_steps = max(total_steps - warmup_steps, 1)

def make_warmup_cosine_scheduler(optimizer):
    warmup_sched = lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    decay_sched = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=1e-6
    )
    return lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps]
    )

scheduler_amplitude = make_warmup_cosine_scheduler(opt_fn_amplitude)
scheduler_phase = make_warmup_cosine_scheduler(opt_fn_phase)
print(f"Schedule: {warmup_steps} warmup steps ({1e-6:.1e} -> {LR:.1e}) + {decay_steps} cosine steps -> 1e-6 (Total: {total_steps} steps across {args.epochs} epochs)")

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
    "epoch_time_sec": [],
}

# Early Stopping Initialization (Decoupled, gated to final 15% of training)
stopped_amplitude = False
stopped_phase = False

if EARLY_STOP > 0:
    min_lr_target = 1e-6
    early_stop_amplitude = EarlyStopping(
        patience=EARLY_STOP, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    early_stop_phase = EarlyStopping(
        patience=EARLY_STOP, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    print(f"Early stopping enabled: patience={EARLY_STOP} (gated to final 15%: epoch >= {int(0.85 * args.epochs)})")
else:
    print("Early stopping disabled.")

print(f"Starting training for {args.epochs} epochs (LR={LR:.1e}, clip_norm={CLIP_NORM}, beta={BETA}, batch_size={BATCH_SIZE}, precond_freq={PRECONDITION_FREQ})...")

for epoch in range(args.epochs):
    t0 = time.time()
    avg_loss_ampl = train_one_epoch_mono(
        model_amplitude, opt_fn_amplitude, scheduler_amplitude, loss_fn_amplitude, device, a_train_loader,
        clip_norm=CLIP_NORM, lambda_latent=lambda_latent
    )
    avg_loss_ph = train_one_epoch_mono(
        model_phase, opt_fn_phase, scheduler_phase, loss_fn_phase, device, p_train_loader,
        clip_norm=CLIP_NORM, lambda_latent=lambda_latent
    )

    # Validate
    avg_vloss_ampl = validate(model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=lambda_latent)
    avg_vloss_ph = validate(model_phase, p_val_loader, loss_fn_phase, device=device, lambda_latent=lambda_latent)

    current_lr_ampl = opt_fn_amplitude.param_groups[0]["lr"]
    current_lr_ph = opt_fn_phase.param_groups[0]["lr"]
    epoch_duration = time.time() - t0

    # Record history
    history["epoch"].append(epoch + 1)
    history["train_loss_amplitude"].append(avg_loss_ampl)
    history["val_loss_amplitude"].append(avg_vloss_ampl)
    history["lr_amplitude"].append(current_lr_ampl)
    history["train_loss_phase"].append(avg_loss_ph)
    history["val_loss_phase"].append(avg_vloss_ph)
    history["lr_phase"].append(current_lr_ph)
    history["epoch_time_sec"].append(epoch_duration)

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

    # Save best Phase model & optimizer state
    if avg_vloss_ph < best_vloss_phase:
        best_vloss_phase = avg_vloss_ph
        torch.save(raw_model_phase.state_dict(), f"{title}_phase_best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model_phase.state_dict(),
                "optimizer_state_dict": opt_fn_phase.state_dict(),
                "scheduler_state_dict": scheduler_phase.state_dict(),
                "best_vloss": best_vloss_phase,
            },
            f"{title}_phase_checkpoint.pt",
        )

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
        print(f"Epoch [{epoch + 1}/{args.epochs}] ({epoch_duration:.2f}s)")
        print(f"  Amplitude | Train Loss: {avg_loss_ampl:.6f} | Val Loss: {avg_vloss_ampl:.6f} | Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}")
        print(f"  Phase     | Train Loss: {avg_loss_ph:.6f} | Val Loss: {avg_vloss_ph:.6f} | Best: {best_vloss_phase:.6f} | LR: {current_lr_ph:.2e}")

    # Check Early Stopping (gated to final 15% of training)
    if EARLY_STOP > 0 and epoch >= int(0.85 * args.epochs):
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
# Save Metrics (CSV and NPY)
# ---------------------------------------------------------------------------
csv_filename = f"{title}_metrics.csv"
try:
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_filename, index=False)
    print(f"\nUnified metrics CSV saved: {csv_filename}")
except Exception as e:
    print(f"\nWarning: Failed to save metrics CSV ({e}).")

npy_filename = f"{title}_metrics.npy"
try:
    np.save(npy_filename, history)
    print(f"Unified metrics NPY saved: {npy_filename}")
except Exception as e:
    print(f"\nWarning: Failed to save metrics NPY ({e}).")

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

plot_phase_filename = f"{title}_phase_plot.png"
plot_training_curves(
    epochs=history["epoch"],
    train_losses=history["train_loss_phase"],
    val_losses=history["val_loss_phase"],
    lrs=history["lr_phase"],
    run_title=title,
    metric_name="Phase",
    output_path=plot_phase_filename,
)

print(f"\nTraining completed successfully for run '{title}'.")

# ---------------------------------------------------------------------------
# Evaluate Best Saved Models and Plot Mismatch Distribution
# ---------------------------------------------------------------------------
best_ampl_path = f"{title}_amplitude_best.pt"
best_phase_path = f"{title}_phase_best.pt"

print(f"\nLoading best saved models for mismatch evaluation...")
raw_model_amplitude.load_state_dict(torch.load(best_ampl_path, map_location=device))
raw_model_phase.load_state_dict(torch.load(best_phase_path, map_location=device))

print("Computing test set waveform mismatches (unnormalized)...")
norm_stats = {
    "mu_ampl": mu_ampl,
    "sd_ampl": sd_ampl,
    "mu_phase": mu_phase,
    "sd_phase": sd_phase,
}
mismatches = compute_mismatches(
    model_amplitude, model_phase, test_loader, device=device, norm=100.0, norm_stats=norm_stats
)

plot_mismatch_filename = f"{title}_mismatch_distribution.png"
plot_mismatch_distribution(
    mismatches=mismatches,
    run_title=title,
    output_path=plot_mismatch_filename,
)

print(f"\nSaved artifacts:")
print(f"  - Best Amplitude Model:      {title}_amplitude_best.pt")
print(f"  - Amplitude Full Checkpoint: {title}_amplitude_checkpoint.pt (includes optimizer state)")
print(f"  - Best Phase Model:          {title}_phase_best.pt")
print(f"  - Phase Full Checkpoint:     {title}_phase_checkpoint.pt (includes optimizer state)")
print(f"  - Normalization NPZ (mu,sd): {norm_npz_path}")
print(f"  - History CSV:               {csv_filename}")
print(f"  - History NPY:               {npy_filename}")
print(f"  - Amplitude Plot:            {plot_ampl_filename}")
print(f"  - Phase Plot:                {plot_phase_filename}")
print(f"  - Mismatch Plot:             {plot_mismatch_filename}")
print(f"  - Test Set Mismatch Stats:   Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
