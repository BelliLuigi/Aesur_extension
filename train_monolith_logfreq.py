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
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

from models import AutoencoderMono
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
parser.add_argument("--epochs", type=int, default=5000, help="Training epochs (default: 5000)")
parser.add_argument("--early_stop", type=int, default=300, help="Early stopping patience in epochs (default: 300, 0 to disable)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
parser.add_argument("--omega_floor", type=float, default=1e-6, help="Floor for instantaneous frequency before log (default: 1e-6)")
args = parser.parse_args()

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
filepath = "datasets/NRSur7dq4v2_dataset.hdf"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

print(f"Loading waveforms and computing log-frequency from '{filepath}'...")
with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    n_samples, sample_len = waveforms_ds.shape
    logfreq = np.empty((n_samples, sample_len), dtype=np.float32)

    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        w = waveforms_ds[start:end]
        ph = wave_to_phase(w)
        logfreq[start:end] = phase_to_logfreq(ph, omega_floor=args.omega_floor)

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

# Save normalization constants
norm_npz_path = f"{title}_logfreq_norm.npz"
np.savez(norm_npz_path, mu=mu, sd=sd, omega_floor=args.omega_floor)
print(f"Saved logfreq normalization constants: {norm_npz_path}")

# Build PyTorch Tensors and DataLoaders
logf_tensor = torch.from_numpy(logfreq)

l_train = Subset(logf_tensor, train_indices)
l_val   = Subset(logf_tensor, val_indices)
l_test  = Subset(logf_tensor, test_indices)

l_train_loader = DataLoader(l_train, batch_size=args.batch_size, shuffle=True)
l_val_loader = DataLoader(l_val, batch_size=2048, shuffle=False)
l_test_loader = DataLoader(l_test, batch_size=2048, shuffle=False)

# ---------------------------------------------------------------------------
# Model, Loss Function & Regularization
# ---------------------------------------------------------------------------
raw_model_logfreq = AutoencoderMono().to(device)

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

loss_fn_logfreq = nn.SmoothL1Loss(beta=1e-3)
lambda_latent = 0.0

# ---------------------------------------------------------------------------
# Optimizer (SOAP) & Scheduler (Plateau)
# ---------------------------------------------------------------------------
opt_fn_logfreq = opt_add.SOAP(
    model_logfreq.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
    weight_decay=0.0, precondition_frequency=40, merge_dims=False, eps=0.0001
)
scheduler_logfreq = lr_scheduler.ReduceLROnPlateau(
    opt_fn_logfreq, factor=0.5, patience=150, threshold=0.003, threshold_mode="rel", min_lr=5e-7, cooldown=50
)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
best_vloss_logfreq = float("inf")

history = {
    "epoch": [],
    "train_loss_logfreq": [],
    "val_loss_logfreq": [],
    "lr_logfreq": [],
}

stopped_logfreq = False

if args.early_stop > 0:
    min_lr_target = 5e-7
    early_stop_logfreq = EarlyStopping(
        patience=args.early_stop, threshold=0.001, mode="rel", min_lr=min_lr_target, patience_at_min_lr=50
    )
    print(f"Early stopping enabled: patience={args.early_stop}, floor_patience=50 at min_lr={min_lr_target:.1e}")
else:
    print("Early stopping disabled.")

print(f"Starting training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    avg_loss_lf = train_one_epoch(
        model_logfreq, opt_fn_logfreq, loss_fn_logfreq, device, epoch, None, l_train_loader, lambda_latent=lambda_latent
    )
    avg_vloss_lf = validate(model_logfreq, l_val_loader, loss_fn_logfreq, device=device, lambda_latent=lambda_latent)
    scheduler_logfreq.step(avg_vloss_lf)
    current_lr_lf = scheduler_logfreq.get_last_lr()[0]

    # Record history
    history["epoch"].append(epoch + 1)
    history["train_loss_logfreq"].append(avg_loss_lf)
    history["val_loss_logfreq"].append(avg_vloss_lf)
    history["lr_logfreq"].append(current_lr_lf)

    # Save best Log-Frequency model & checkpoint
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
        print(f"Epoch [{epoch + 1}/{args.epochs}] | Train Loss: {avg_loss_lf:.6f} | Val Loss: {avg_vloss_lf:.6f} | Best: {best_vloss_logfreq:.6f} | LR: {current_lr_lf:.2e}")

    # Check Early Stopping
    if args.early_stop > 0:
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

# Compute test set evaluation loss
best_logf_path = f"{title}_logfreq_best.pt"
raw_model_logfreq.load_state_dict(torch.load(best_logf_path, map_location=device))
test_loss = validate(model_logfreq, l_test_loader, loss_fn_logfreq, device=device, lambda_latent=lambda_latent)

print(f"\nTraining completed successfully for run '{title}'.")
print(f"  - Test Loss (Best Model):    {test_loss:.6f}")
print(f"  - Best LogFreq Model:        {best_logf_path}")
print(f"  - LogFreq Checkpoint:        {title}_logfreq_checkpoint.pt")
print(f"  - Normalization NPZ (mu,sd): {norm_npz_path}")
print(f"  - Metrics CSV:               {csv_filename}")
print(f"  - Metrics NPY:               {npy_filename}")
print(f"  - Loss Curves Plot:          {plot_logf_filename}")
