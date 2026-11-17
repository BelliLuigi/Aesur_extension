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
    train_one_epoch,
    validate,
    plot_training_curves,
    EarlyStopping,
)


def wave_to_phase(w: np.ndarray) -> np.ndarray:
    """
    Computes unwrapped zero-aligned phase in radians (TasosAE_LOGFREQ approach).
    w: complex array of shape (N, L)
    returns: phase array of shape (N, L) where phase[:, 0] == 0
    """
    ph = -np.unwrap(np.angle(w), axis=-1)
    return ph - ph[:, :1]


def phase_to_logfreq(ph: np.ndarray, omega_floor: float = 1e-6) -> np.ndarray:
    """
    Computes instantaneous log-frequency from phase trajectory:
      omega[k] = phi[k+1] - phi[k] for k in 0..L-2, with last value repeated
      logfreq = log(max(omega, omega_floor))
    """
    om = np.diff(ph, axis=-1)
    om = np.concatenate([om, om[:, -1:]], axis=-1)
    om = np.maximum(om, omega_floor)
    return np.log(om).astype(np.float32)


def to_phase_numpy(x_norm: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Reconstructs unwrapped phase in numpy:
      omega = exp(x_norm * sd + mu)
      ph = [0, cumsum(omega[:-1])]
    """
    om = np.exp(x_norm.astype(np.float64) * sd + mu)
    ph_cum = np.cumsum(om[:, :-1], axis=-1)
    zeros = np.zeros((len(om), 1), dtype=np.float64)
    return np.concatenate([zeros, ph_cum], axis=-1)


# ---------------------------------------------------------------------------
# CLI Arguments (Minimal Execution Controls)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train AutoencoderMono on Log-frequency (SOAP)")
parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
parser.add_argument("--epochs", type=int, default=800, help="Training epochs (default: 800)")
parser.add_argument("--small", action=argparse.BooleanOptionalAction, default=False, help="Use small dataset variant (NRSur7dq4v2_dataset_small.hdf) (default: False)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable torch.compile (default: False)")
parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable TF32 precision on CUDA (default: False)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Hardcoded Hyperparameters & Physical Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
OMEGA_FLOOR = 1e-6
PRECONDITION_FREQ = 15
BETA = 0.3
CLIP_NORM = 1.0
CLAMP_TARGET = 4.0
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
        title = f"AE_mono_logfreq_soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_name = "AutoencoderMono"
print(f"=== Starting Run: {title} | Model: {model_name} | Representation: LogFreq | Optimizer: SOAP ===")

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
# Load Dataset from HDF5 and Extract Log-Frequency Only
# ---------------------------------------------------------------------------
filepath = "datasets/NRSur7dq4v2_dataset_small.hdf" if args.small else "datasets/NRSur7dq4v2_dataset.hdf"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

print(f"Loading waveforms and computing log-frequency from '{filepath}' (small={args.small})...")
with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    total_avail, sample_len = waveforms_ds.shape
    n_samples = total_avail
    logfreq = np.empty((n_samples, sample_len), dtype=np.float32)

    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        w = waveforms_ds[start:end]
        ph = wave_to_phase(w)
        logfreq[start:end] = phase_to_logfreq(ph, omega_floor=OMEGA_FLOOR)

print(f"Loaded and converted {n_samples} waveforms to log-frequency (length: {sample_len}).")

# ---------------------------------------------------------------------------
# Split Dataset (80% train / 10% val / 10% test)
# ---------------------------------------------------------------------------
idx_all = np.arange(n_samples)
gen = torch.Generator().manual_seed(42)
j_train_idx, j_val_idx, j_test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)

train_indices = np.array(j_train_idx.indices)
val_indices = np.array(j_val_idx.indices)
test_indices = np.array(j_test_idx.indices)

print(f"Dataset split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

# Compute per-timestep mean and standard deviation strictly on the training set
print("Normalizing log-frequency using train-set per-timestep statistics...")
mu = np.zeros(sample_len, dtype=np.float64)
sq = np.zeros(sample_len, dtype=np.float64)

for s in range(0, len(train_indices), 5000):
    blk = logfreq[np.sort(train_indices[s:s + 5000])].astype(np.float64)
    mu += blk.sum(axis=0)
    sq += (blk ** 2).sum(axis=0)

mu /= len(train_indices)
sd = np.sqrt(np.maximum(sq / len(train_indices) - mu ** 2, 0.0)) + 1e-6

# Normalize in-place across all samples
for s in range(0, n_samples, 5000):
    logfreq[s:s + 5000] = (logfreq[s:s + 5000] - mu) / sd

# Fix 1: Clamp normalized log-frequency to prevent -37 sigma spikes from omega_floor
if CLAMP_TARGET > 0:
    n_clamped = int((np.abs(logfreq) > CLAMP_TARGET).sum())
    logfreq = np.clip(logfreq, -CLAMP_TARGET, CLAMP_TARGET)
    print(f"Fix 1 applied: clamped {n_clamped} outlier values to [-{CLAMP_TARGET}, {CLAMP_TARGET}] ({100 * n_clamped / logfreq.size:.3f}%).")

# Save normalization constants
norm_npz_path = f"{title}_logfreq_norm.npz"
np.savez(norm_npz_path, mu=mu, sd=sd, omega_floor=OMEGA_FLOOR)
print(f"Saved logfreq normalization constants: {norm_npz_path}")

# Build PyTorch Tensors and DataLoaders
logf_tensor = torch.from_numpy(logfreq)

l_train = Subset(logf_tensor, train_indices)
l_val   = Subset(logf_tensor, val_indices)
l_test  = Subset(logf_tensor, test_indices)

l_train_loader = DataLoader(l_train, batch_size=BATCH_SIZE, shuffle=True)
l_val_loader = DataLoader(l_val, batch_size=2048, shuffle=False)
l_test_loader = DataLoader(l_test, batch_size=2048, shuffle=False)

# ---------------------------------------------------------------------------
# Model, Loss Function & Regularization
# ---------------------------------------------------------------------------
raw_model_logfreq = AutoencoderMono().to(device)
init_kaiming_weights(raw_model_logfreq)

if args.compile:
    if sys.version_info >= (3, 13):
        print("Notice: torch.compile/Dynamo is not supported on Python 3.13+. Continuing with uncompiled model.")
        model_logfreq = raw_model_logfreq
    else:
        try:
            print("Compiling model with torch.compile...")
            model_logfreq = torch.compile(raw_model_logfreq)
        except Exception as e:
            print(f"Notice: torch.compile failed ({e}). Continuing with uncompiled model.")
            model_logfreq = raw_model_logfreq
else:
    print("torch.compile disabled. Running uncompiled eager model.")
    model_logfreq = raw_model_logfreq

loss_fn_logfreq = nn.SmoothL1Loss(beta=BETA)
lambda_latent = 0.0

# ---------------------------------------------------------------------------
# Training & Validation Functions
# ---------------------------------------------------------------------------
def train_one_epoch_logfreq(
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


def validate_logfreq(
    model, loader, loss_fn, device, mu_t, sd_t, lambda_latent: float = 0.0
):
    model.eval()
    total_loss = 0.0
    total_phase_err = 0.0
    n_batches = 0
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device=device, dtype=dtype)
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)

            if lambda_latent > 0.0:
                outputs, z = model(inputs, return_latent=True)
                recon_loss = loss_fn(outputs, inputs)
                loss = recon_loss + lambda_latent * torch.mean(z ** 2)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, inputs)

            total_loss += loss.item()

            # Denormalize log-frequency and integrate phase
            out_2d = outputs.squeeze(1)
            inp_2d = inputs.squeeze(1)
            om_pred = torch.exp(out_2d.double() * sd_t + mu_t)
            om_true = torch.exp(inp_2d.double() * sd_t + mu_t)

            # Cumulative sum over indices 0..L-2 (excluding the duplicated final timestep L-1)
            ph_pred_cum = torch.cumsum(om_pred[:, :-1], dim=-1)
            ph_true_cum = torch.cumsum(om_true[:, :-1], dim=-1)

            # Mean absolute unwrapped phase error in radians
            phase_err = torch.mean(torch.abs(ph_pred_cum - ph_true_cum)).item()
            total_phase_err += phase_err
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_phase_err = total_phase_err / max(n_batches, 1)
    return avg_loss, avg_phase_err


# ---------------------------------------------------------------------------
# Optimizer (SOAP) & Scheduler (Warmup + Cosine Decay)
# ---------------------------------------------------------------------------
opt_fn_logfreq = opt_add.SOAP(
    model_logfreq.parameters(),
    lr=LR,
    betas=(0.95, 0.95),
    shampoo_beta=0.99,
    weight_decay=0.0,
    precondition_frequency=PRECONDITION_FREQ,
    merge_dims=False,
    eps=1e-8,
)

# Step-based Warmup + Cosine Decay Schedule
total_steps = args.epochs * len(l_train_loader)
warmup_steps = min(WARMUP_STEPS, max(1, total_steps // 2))
decay_steps = max(total_steps - warmup_steps, 1)

warmup_sched = lr_scheduler.LinearLR(
    opt_fn_logfreq, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
)
decay_sched = lr_scheduler.CosineAnnealingLR(
    opt_fn_logfreq, T_max=decay_steps, eta_min=1e-6
)
scheduler_logfreq = lr_scheduler.SequentialLR(
    opt_fn_logfreq, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps]
)
print(f"Schedule: {warmup_steps} warmup steps ({1e-6:.1e} -> {LR:.1e}) + {decay_steps} cosine steps -> 1e-6 (Total: {total_steps} steps across {args.epochs} epochs)")

# Normalization tensors on device for fast validation phase evaluation
mu_tensor = torch.from_numpy(mu).to(device=device, dtype=torch.float64)
sd_tensor = torch.from_numpy(sd).to(device=device, dtype=torch.float64)

# Early Stopping (Decoupled, gated to final 15% of training)
stopped_logfreq = False
if EARLY_STOP > 0:
    min_lr_target = 1e-6
    early_stop_logfreq = EarlyStopping(
        patience=EARLY_STOP, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    print(f"Early stopping enabled: patience={EARLY_STOP} (gated to final 15%: epoch >= {int(0.85 * args.epochs)})")
else:
    print("Early stopping disabled.")

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
best_vloss_logfreq = float("inf")
best_vphase_err = float("inf")

history = {
    "epoch": [],
    "train_loss_logfreq": [],
    "val_loss_logfreq": [],
    "val_phase_err": [],
    "lr_logfreq": [],
    "epoch_time_sec": [],
}

print(f"Starting training for {args.epochs} epochs (LR={LR:.1e}, clip_norm={CLIP_NORM}, clamp={CLAMP_TARGET}, beta={BETA}, batch_size={BATCH_SIZE}, precond_freq={PRECONDITION_FREQ})...")

for epoch in range(args.epochs):
    t0 = time.time()
    avg_loss_lf = train_one_epoch_logfreq(
        model_logfreq, opt_fn_logfreq, scheduler_logfreq, loss_fn_logfreq, device, l_train_loader,
        clip_norm=CLIP_NORM, lambda_latent=lambda_latent
    )
    avg_vloss_lf, avg_phase_err = validate_logfreq(
        model_logfreq, l_val_loader, loss_fn_logfreq, device=device, mu_t=mu_tensor, sd_t=sd_tensor, lambda_latent=lambda_latent
    )
    current_lr_lf = opt_fn_logfreq.param_groups[0]["lr"]
    epoch_duration = time.time() - t0

    # Record history
    history["epoch"].append(epoch + 1)
    history["train_loss_logfreq"].append(avg_loss_lf)
    history["val_loss_logfreq"].append(avg_vloss_lf)
    history["val_phase_err"].append(avg_phase_err)
    history["lr_logfreq"].append(current_lr_lf)
    history["epoch_time_sec"].append(epoch_duration)

    # Save best Log-Frequency model & checkpoint (by validation loss)
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
                "best_vphase_err": best_vphase_err,
            },
            f"{title}_logfreq_checkpoint.pt",
        )

    # Save best Phase model (by cumulative unwrapped phase error in radians)
    if avg_phase_err < best_vphase_err:
        best_vphase_err = avg_phase_err
        torch.save(raw_model_logfreq.state_dict(), f"{title}_phase_best.pt")

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] ({epoch_duration:.2f}s) | "
            f"Train Loss: {avg_loss_lf:.6f} | Val Loss: {avg_vloss_lf:.6f} | "
            f"Val Phase Err: {avg_phase_err:.4f} rad | "
            f"Best Loss: {best_vloss_logfreq:.6f} | Best Phase: {best_vphase_err:.4f} rad | "
            f"LR: {current_lr_lf:.2e}"
        )

    # Check Early Stopping (gated to final 15% of training)
    if EARLY_STOP > 0 and epoch >= int(0.85 * args.epochs):
        stopped_logfreq = early_stop_logfreq.step(avg_vloss_lf, current_lr_lf)
        if stopped_logfreq:
            print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
            print(f"  Reason: {early_stop_logfreq.stop_reason}")
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
# Generate Training Loss and Validation Loss Plot
# ---------------------------------------------------------------------------
plot_logf_filename = f"{title}_logfreq_plot.png"
plot_training_curves(
    epochs=history["epoch"],
    train_losses=history["train_loss_logfreq"],
    val_losses=history["val_loss_logfreq"],
    lrs=history["lr_logfreq"],
    run_title=title,
    metric_name="LogFreq",
    output_path=plot_logf_filename,
)

# Compute test set evaluation loss and phase error
best_logf_path = f"{title}_logfreq_best.pt"
best_phase_path = f"{title}_phase_best.pt"

raw_model_logfreq.load_state_dict(torch.load(best_logf_path, map_location=device))
test_loss, test_phase_err = validate_logfreq(
    model_logfreq, l_test_loader, loss_fn_logfreq, device=device, mu_t=mu_tensor, sd_t=sd_tensor, lambda_latent=lambda_latent
)

raw_model_logfreq.load_state_dict(torch.load(best_phase_path, map_location=device))
test_phase_model_loss, test_phase_model_err = validate_logfreq(
    model_logfreq, l_test_loader, loss_fn_logfreq, device=device, mu_t=mu_tensor, sd_t=sd_tensor, lambda_latent=lambda_latent
)

print(f"\nTraining completed successfully for run '{title}'.")
print(f"  - Test Loss (Best Loss Model):        {test_loss:.6f} (Phase Err: {test_phase_err:.4f} rad)")
print(f"  - Test Phase Err (Best Phase Model):    {test_phase_model_err:.4f} rad (Loss: {test_phase_model_loss:.6f})")
print(f"  - Best LogFreq Model:                 {best_logf_path}")
print(f"  - Best Phase Model:                   {best_phase_path}")
print(f"  - LogFreq Checkpoint:                 {title}_logfreq_checkpoint.pt")
print(f"  - Normalization NPZ (mu,sd):          {norm_npz_path}")
print(f"  - Metrics CSV:                        {csv_filename}")
print(f"  - Metrics NPY:                        {npy_filename}")
print(f"  - Loss Curves Plot:                   {plot_logf_filename}")
