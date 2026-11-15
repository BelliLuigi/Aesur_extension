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

from models import CoorbitalAE, FlexE, FlexD, init_kaiming_weights
from utils import (
    plot_coorbital_training_curves,
    TasosOverlapTorch,
    plot_mismatch_distribution,
    EarlyStopping,
    reconstruct_inertial_waveform,
    compute_inertial_mismatches,
    plot_inertial_mismatch_ablation,
)

# ---------------------------------------------------------------------------
# Section 0: Phase & Log-Frequency Helper Functions (TasosAE_LOGFREQ approach)
# ---------------------------------------------------------------------------
def wave_to_phase(w: np.ndarray) -> np.ndarray:
    """
    Computes unwrapped zero-aligned phase in physical radians.
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
    Reconstructs unwrapped phase from normalized log-frequency:
      omega = exp(x_norm * sd + mu)
      ph = [0, cumsum(omega[:-1])]
    """
    om = np.exp(x_norm.astype(np.float64) * sd + mu)
    ph_cum = np.cumsum(om[:, :-1], axis=-1)
    zeros = np.zeros((len(om), 1), dtype=np.float64)
    return np.concatenate([zeros, ph_cum], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Section 1: Wrapper Class embedding CoorbitalAE + Detached LogFreq Module
# ---------------------------------------------------------------------------
class CoorbitalAEWithLogFreq(nn.Module):
    """
    Wrapper integrating CoorbitalAE alongside a detached log-frequency autoencoder
    using FlexE and FlexD modules without modifying models.py or utils.py.
    """
    def __init__(self, coorbital_model: CoorbitalAE = None, logfreq_btnk: int = 4, logfreq_width: float = 0.75):
        super().__init__()
        self.coorbital = coorbital_model if coorbital_model is not None else CoorbitalAE()
        self.E_logfreq = FlexE(bottleneck=logfreq_btnk, width=logfreq_width, channels=1)
        self.D_logfreq = FlexD(bottleneck=logfreq_btnk, width=logfreq_width, channels=1)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if '_modules' in self.__dict__ and 'coorbital' in self._modules:
                return getattr(self._modules['coorbital'], name)
            raise

    def forward(self, ophase, q, chi1, chi2, cooA, cooP, logfreq=None, return_latent: bool = False):
        coo_recs = self.coorbital(ophase, q, chi1, chi2, cooA, cooP, return_latent=return_latent)
        if logfreq is not None:
            z_lf = self.E_logfreq(logfreq)
            rec_lf = self.D_logfreq(z_lf)
            if return_latent:
                recs, latents = coo_recs
                return recs + (rec_lf,), latents + (z_lf,)
            else:
                return coo_recs + (rec_lf,)
        return coo_recs


# Component identifiers
COO_COMPONENTS = ["ophase", "q", "chi1", "chi2", "cooA", "cooP"]
ALL_COMPONENTS = ["ophase", "q", "chi1", "chi2", "cooA", "cooP", "logfreq"]

# ==============================================================================
# Hyperparameters & Configuration
# ==============================================================================
DATASET_PATH     = "datasets/NRSur7dq4v2_dataset.hdf"
SAMPLES          = None         # None to use all available samples (e.g. 102400)
EPOCHS           = 5000         # Total training epochs
BATCH_SIZE       = 128          # Training batch size
VAL_BATCH_SIZE   = 2048         # Validation & test batch size
LR               = 1e-3         # Initial learning rate for SOAP (reduced from 5e-3 with eps=1e-8)
MIN_LR           = 5e-7         # Minimum learning rate floor
LR_PATIENCE      = 150          # ReduceLROnPlateau patience (epochs)
EARLY_STOP       = 300          # Early stopping patience (epochs, 0 to disable)
LAMBDA_LATENT    = 0.0          # L2 latent regularization weight
OMEGA_FLOOR      = 1e-6         # Floor for frequency derivation

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train CoorbitalAE with normalized phases and detached LogFreq autoencoder (SOAP optimizer)"
)
parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of samples to load (default: all)")
parser.add_argument("--early_stop", type=int, default=EARLY_STOP, help="Early stopping patience (default: 300)")
parser.add_argument("--lr", type=float, default=LR, help="Initial learning rate (default: 1e-3)")
parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clipping norm threshold (default: 1.0, 0 to disable)")
parser.add_argument("--clamp_logfreq", type=float, default=4.0, help="Symmetric clamp threshold for normalized logfreq (default: 4.0, 0 to disable)")
parser.add_argument("--clamp_coop", type=float, default=6.0, help="Symmetric clamp threshold for normalized cooP (default: 6.0, 0 to disable)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"CoorbitalAE_logfreq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SAMPLES = args.samples
EARLY_STOP = args.early_stop
LR = args.lr

model_name = "CoorbitalAEWithLogFreq"
print(f"=== Starting Run: {title} | Model: {model_name} | Optimizer: SOAP | Phase: Normalized | Epochs: {EPOCHS} ===")

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
filepath = DATASET_PATH
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at '{filepath}'")

print(f"Loading coorbital kinematics, dynamics, and waveforms from '{filepath}'...")
with h5py.File(filepath, "r") as f:
    total_avail = f["Waveforms_coorb"].shape[0]
    n_samples = min(SAMPLES, total_avail) if (SAMPLES and SAMPLES > 0) else total_avail
    sample_len = f["Waveforms_coorb"].shape[1]

    ophase_arr  = np.empty((n_samples, 1, sample_len), dtype=np.float32)
    q_arr       = np.empty((n_samples, 4, sample_len), dtype=np.float32)
    chi1_arr    = np.empty((n_samples, 3, sample_len), dtype=np.float32)
    chi2_arr    = np.empty((n_samples, 3, sample_len), dtype=np.float32)
    cooA_arr    = np.empty((n_samples, 1, sample_len), dtype=np.float32)
    cooP_arr    = np.empty((n_samples, 1, sample_len), dtype=np.float32)
    logfreq_arr = np.empty((n_samples, 1, sample_len), dtype=np.float32)

    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        cur_n = end - start

        # 1. Orbital Phase in physical radians (unnormalized)
        ophase_arr[start:end, 0, :] = f["Dynamics/orbphase"][start:end].astype(np.float32)

        # 2. Coprecessing Quaternions (N, 4, 2048)
        q_arr[start:end] = f["Dynamics/q_copr"][start:end].astype(np.float32)

        # 3. Dynamic spins chiA (N, 2048, 3) -> transpose to (N, 3, 2048)
        chi1_arr[start:end] = np.transpose(f["Dynamics/chiA"][start:end].astype(np.float32), (0, 2, 1))

        # 4. Dynamic spins chiB (N, 2048, 3) -> transpose to (N, 3, 2048)
        chi2_arr[start:end] = np.transpose(f["Dynamics/chiB"][start:end].astype(np.float32), (0, 2, 1))

        # 5 & 6. Co-orbital Waveform (2,2): Amplitude and Unwrapped Phase in physical radians (no phase/100)
        wfs_coorb = f["Waveforms_coorb"][start:end]
        cooA_arr[start:end, 0, :] = np.abs(wfs_coorb).astype(np.float32)
        cooP_arr[start:end, 0, :] = wave_to_phase(wfs_coorb).astype(np.float32)

        # 7. Log-frequency from waveform phase
        wfs_for_logf = f["Waveforms"][start:end] if "Waveforms" in f else wfs_coorb
        ph_for_logf = wave_to_phase(wfs_for_logf)
        logfreq_arr[start:end, 0, :] = phase_to_logfreq(ph_for_logf, omega_floor=OMEGA_FLOOR)

        if (end % 20000 == 0) or (end == n_samples):
            print(f"  Loaded {end}/{n_samples} samples ({end / n_samples * 100:.1f}%)")

# ---------------------------------------------------------------------------
# Split Dataset & Compute Training-Set Normalization Statistics
# ---------------------------------------------------------------------------
idx_all = np.arange(n_samples)
gen = torch.Generator().manual_seed(42)
j_train_idx, j_val_idx, j_test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)

train_indices = np.array(j_train_idx.indices)
val_indices = np.array(j_val_idx.indices)
test_indices = np.array(j_test_idx.indices)

print(f"Dataset split: Train={len(train_indices)} | Val={len(val_indices)} | Test={len(test_indices)}")

# 1. Normalize ophase using train-set per-timestep mean and standard deviation
mu_ophase = ophase_arr[train_indices, 0, :].mean(axis=0)
std_ophase = ophase_arr[train_indices, 0, :].std(axis=0) + 1e-6
ophase_arr[:, 0, :] = (ophase_arr[:, 0, :] - mu_ophase) / std_ophase

# 2. Normalize cooP using train-set per-timestep mean and standard deviation
mu_cooP = cooP_arr[train_indices, 0, :].mean(axis=0)
std_cooP = cooP_arr[train_indices, 0, :].std(axis=0) + 1e-6
cooP_arr[:, 0, :] = (cooP_arr[:, 0, :] - mu_cooP) / std_cooP
if args.clamp_coop > 0:
    n_clamped_coop = int((np.abs(cooP_arr) > args.clamp_coop).sum())
    cooP_arr = np.clip(cooP_arr, -args.clamp_coop, args.clamp_coop)
    print(f"Clamped {n_clamped_coop} cooP outliers to [-{args.clamp_coop}, {args.clamp_coop}] ({100 * n_clamped_coop / cooP_arr.size:.3f}%).")

# 3. Normalize cooA using train-set per-timestep mean and standard deviation
mu_cooA = cooA_arr[train_indices, 0, :].mean(axis=0)
sd_cooA = cooA_arr[train_indices, 0, :].std(axis=0) + 1e-6
cooA_arr[:, 0, :] = (cooA_arr[:, 0, :] - mu_cooA) / sd_cooA

# 4. Normalize logfreq using train-set per-timestep mean and standard deviation
mu_logf = logfreq_arr[train_indices, 0, :].mean(axis=0)
sd_logf = logfreq_arr[train_indices, 0, :].std(axis=0) + 1e-6
logfreq_arr[:, 0, :] = (logfreq_arr[:, 0, :] - mu_logf) / sd_logf
if args.clamp_logfreq > 0:
    n_clamped = int((np.abs(logfreq_arr[:, 0, :]) > args.clamp_logfreq).sum())
    logfreq_arr[:, 0, :] = np.clip(logfreq_arr[:, 0, :], -args.clamp_logfreq, args.clamp_logfreq)
    print(f"Clamped {n_clamped} logfreq outliers to [-{args.clamp_logfreq}, {args.clamp_logfreq}] ({100 * n_clamped / logfreq_arr.size:.3f}%).")

print(f"Per-timestep normalization applied to ophase, cooP, cooA, and logfreq.")

# Save normalization constants
norm_file = f"{title}_norm_params.npz"
np.savez(
    norm_file,
    mu_ophase=mu_ophase,
    std_ophase=std_ophase,
    mu_cooP=mu_cooP,
    std_cooP=std_cooP,
    mu_cooA=mu_cooA,
    sd_cooA=sd_cooA,
    mu_logf=mu_logf,
    sd_logf=sd_logf,
    omega_floor=OMEGA_FLOOR,
)
print(f"Saved normalization constants to: {norm_file}")

# Build Tensors and DataLoaders
joint_dataset = TensorDataset(
    torch.from_numpy(ophase_arr),
    torch.from_numpy(q_arr),
    torch.from_numpy(chi1_arr),
    torch.from_numpy(chi2_arr),
    torch.from_numpy(cooA_arr),
    torch.from_numpy(cooP_arr),
    torch.from_numpy(logfreq_arr),
)
del ophase_arr, q_arr, chi1_arr, chi2_arr, cooA_arr, cooP_arr, logfreq_arr

j_train = Subset(joint_dataset, train_indices)
j_val   = Subset(joint_dataset, val_indices)
j_test  = Subset(joint_dataset, test_indices)

train_loader = DataLoader(j_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available(), drop_last=True)
val_loader   = DataLoader(j_val, batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(j_test, batch_size=VAL_BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# Model, Loss Functions & Regularization
# ---------------------------------------------------------------------------
raw_model = CoorbitalAEWithLogFreq().to(device)
init_kaiming_weights(raw_model)

total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
coo_params = sum(p.numel() for p in raw_model.coorbital.parameters() if p.requires_grad)
lf_params = sum(p.numel() for p in raw_model.E_logfreq.parameters()) + sum(p.numel() for p in raw_model.D_logfreq.parameters())
print(f"Initialized {model_name} with {total_params:,} trainable parameters (CoorbitalAE: {coo_params:,}, Detached LogFreq: {lf_params:,}).")

if sys.version_info >= (3, 13):
    print("Notice: torch.compile/Dynamo is not supported on Python 3.13+. Continuing with uncompiled models.")
    model = raw_model
else:
    try:
        print("Compiling model with torch.compile...")
        model = torch.compile(raw_model)
    except Exception as e:
        print(f"Notice: torch.compile failed ({e}). Continuing with uncompiled model.")
        model = raw_model

# Smooth L1 loss per branch (beta=1e-3)
loss_fns = {comp: nn.SmoothL1Loss(beta=1e-3) for comp in ALL_COMPONENTS}
lambda_latent = LAMBDA_LATENT

# ---------------------------------------------------------------------------
# Optimizers (SOAP) & Schedulers (Plateau) per component
# ---------------------------------------------------------------------------
optimizers = {
    comp: opt_add.SOAP(
        list(getattr(raw_model, f"E_{comp}").parameters()) + list(getattr(raw_model, f"D_{comp}").parameters()),
        lr=LR,
        betas=(0.95, 0.999),
        shampoo_beta=0.99,
        weight_decay=0.0,
        precondition_frequency=40,
        merge_dims=False,
        eps=1e-8,
    )
    for comp in ALL_COMPONENTS
}

schedulers = {
    comp: lr_scheduler.ReduceLROnPlateau(
        optimizers[comp],
        factor=0.5,
        patience=LR_PATIENCE,
        threshold=0.005,
        threshold_mode="rel",
        min_lr=MIN_LR,
        cooldown=25,
    )
    for comp in ALL_COMPONENTS
}

# ---------------------------------------------------------------------------
# Early Stopping Initialization
# ---------------------------------------------------------------------------
stopped = {comp: False for comp in ALL_COMPONENTS}
best_vloss = {comp: float("inf") for comp in ALL_COMPONENTS}

if EARLY_STOP > 0:
    early_stoppers = {
        comp: EarlyStopping(
            patience=EARLY_STOP,
            threshold=0.001,
            mode="rel",
            min_lr=MIN_LR,
            patience_at_min_lr=50,
        )
        for comp in ALL_COMPONENTS
    }
    print(
        f"Early stopping enabled for all 7 branches: patience={EARLY_STOP}, "
        f"floor_patience=50 at min_lr={MIN_LR:.1e} (rel threshold=0.1%)"
    )
else:
    early_stoppers = {}
    print("Early stopping disabled.")

history = {"epoch": []}
for comp in ALL_COMPONENTS:
    history[f"train_loss_{comp}"] = []
    history[f"val_loss_{comp}"] = []
    history[f"lr_{comp}"] = []

# Keep optimal weights composed across the best epoch of each branch
best_coorbital_state = {k: v.clone() for k, v in raw_model.coorbital.state_dict().items()}
best_full_state = {k: v.clone() for k, v in raw_model.state_dict().items()}

print(f"\nStarting {model_name} training for {EPOCHS} epochs...\n")

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss_sums = {comp: 0.0 for comp in ALL_COMPONENTS}

    for batch in train_loader:
        b_inputs = [x.to(device=device, dtype=torch.float32, non_blocking=True) for x in batch]
        # b_inputs order: ophase, q, chi1, chi2, cooA, cooP, logfreq

        for comp in ALL_COMPONENTS:
            if not stopped[comp]:
                optimizers[comp].zero_grad()

        recs, latents = model(*b_inputs[:6], logfreq=b_inputs[6], return_latent=True)

        total_active_loss = torch.tensor(0.0, device=device)
        for i, comp in enumerate(ALL_COMPONENTS):
            loss_c = loss_fns[comp](recs[i], b_inputs[i])
            if lambda_latent > 0.0:
                loss_c = loss_c + lambda_latent * torch.mean(latents[i] ** 2)

            train_loss_sums[comp] += loss_c.item()

            if not stopped[comp]:
                total_active_loss = total_active_loss + loss_c

        if total_active_loss.requires_grad:
            total_active_loss.backward()
            for comp in ALL_COMPONENTS:
                if not stopped[comp]:
                    if args.clip_norm > 0:
                        params = list(getattr(raw_model, f"E_{comp}").parameters()) + list(getattr(raw_model, f"D_{comp}").parameters())
                        torch.nn.utils.clip_grad_norm_(params, max_norm=args.clip_norm)
                    optimizers[comp].step()

    n_train_batches = max(len(train_loader), 1)
    avg_train_losses = {comp: train_loss_sums[comp] / n_train_batches for comp in ALL_COMPONENTS}

    # Validation
    model.eval()
    val_loss_sums = {comp: 0.0 for comp in ALL_COMPONENTS}
    with torch.no_grad():
        for vbatch in val_loader:
            vb_inputs = [x.to(device=device, dtype=torch.float32, non_blocking=True) for x in vbatch]
            vrecs, vz = model(*vb_inputs[:6], logfreq=vb_inputs[6], return_latent=True)
            for i, comp in enumerate(ALL_COMPONENTS):
                vloss_c = loss_fns[comp](vrecs[i], vb_inputs[i])
                if lambda_latent > 0.0:
                    vloss_c = vloss_c + lambda_latent * torch.mean(vz[i] ** 2)
                val_loss_sums[comp] += vloss_c.item()

    n_val_batches = max(len(val_loader), 1)
    avg_val_losses = {comp: val_loss_sums[comp] / n_val_batches for comp in ALL_COMPONENTS}

    # Step LR Schedulers
    current_lrs = {}
    for comp in ALL_COMPONENTS:
        schedulers[comp].step(avg_val_losses[comp])
        current_lrs[comp] = schedulers[comp].get_last_lr()[0]

    # Record History
    history["epoch"].append(epoch + 1)
    for comp in ALL_COMPONENTS:
        history[f"train_loss_{comp}"].append(avg_train_losses[comp])
        history[f"val_loss_{comp}"].append(avg_val_losses[comp])
        history[f"lr_{comp}"].append(current_lrs[comp])

    # Check and save best per-branch models and update composite best full model
    any_coo_improved = False
    for comp in ALL_COMPONENTS:
        if avg_val_losses[comp] < best_vloss[comp]:
            best_vloss[comp] = avg_val_losses[comp]

            # Save standalone sub-autoencoder state for this component
            sub_state = {
                f"E_{comp}": getattr(raw_model, f"E_{comp}").state_dict(),
                f"D_{comp}": getattr(raw_model, f"D_{comp}").state_dict(),
            }
            torch.save(sub_state, f"{title}_{comp}_best.pt")

            # Update composite optimal states
            if comp in COO_COMPONENTS:
                any_coo_improved = True
                for k, v in raw_model.coorbital.state_dict().items():
                    if k.startswith(f"E_{comp}.") or k.startswith(f"D_{comp}."):
                        best_coorbital_state[k] = v.clone()

            for k, v in raw_model.state_dict().items():
                if k.startswith(f"E_{comp}.") or k.startswith(f"D_{comp}.") or k.startswith(f"coorbital.E_{comp}.") or k.startswith(f"coorbital.D_{comp}."):
                    best_full_state[k] = v.clone()

    if any_coo_improved:
        torch.save(best_coorbital_state, f"{title}_coorbital_best.pt")
    torch.save(best_full_state, f"{title}_wrapper_best.pt")

    # Save periodic full training checkpoint
    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "best_full_state_dict": best_full_state,
                "optimizers": {comp: optimizers[comp].state_dict() for comp in ALL_COMPONENTS},
                "schedulers": {comp: schedulers[comp].state_dict() for comp in ALL_COMPONENTS},
                "best_vloss": best_vloss,
            },
            f"{title}_checkpoint.pt",
        )

    # Logging
    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == EPOCHS:
        print(f"--- Epoch [{epoch + 1}/{EPOCHS}] ---")
        for comp in ALL_COMPONENTS:
            status = "STOPPED" if stopped[comp] else f"LR: {current_lrs[comp]:.2e}"
            print(
                f"  {comp:8s} | Train Loss: {avg_train_losses[comp]:.6f} | "
                f"Val Loss: {avg_val_losses[comp]:.6f} | Best: {best_vloss[comp]:.6f} | {status}"
            )

    # Check Early Stopping
    if EARLY_STOP > 0:
        for comp in ALL_COMPONENTS:
            if not stopped[comp]:
                stopped[comp] = early_stoppers[comp].step(avg_val_losses[comp], current_lrs[comp])

        if all(stopped.values()):
            print(f"\n=== [Early Stopping Triggered for all branches at Epoch {epoch + 1}] ===")
            for comp in ALL_COMPONENTS:
                print(f"  {comp:8s}: {early_stoppers[comp].stop_reason}")
            break

# ---------------------------------------------------------------------------
# Save Unified Metrics (CSV and NPY)
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
# Generate Training Curve Plots
# ---------------------------------------------------------------------------
print("\nGenerating coorbital training curves (2x3 grid)...")
plot_curves_filename = f"{title}_training_curves.png"
plot_coorbital_training_curves(
    history=history,
    component_names=COO_COMPONENTS,
    run_title=f"{title} (Coorbital)",
    output_path=plot_curves_filename,
)

# Plot detached logfreq training curve separately
logfreq_curve_filename = f"{title}_logfreq_curve.png"
try:
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax2 = ax1.twinx()
    ep_list = history["epoch"]
    ax1.plot(ep_list, history["train_loss_logfreq"], label="Train Loss", color="#1f77b4", lw=1.2)
    ax1.plot(ep_list, history["val_loss_logfreq"], label="Val Loss", color="#d62728", lw=1.2)
    ax2.plot(ep_list, history["lr_logfreq"], label="LR", color="black", linestyle="--", lw=1.0)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Smooth L1 Loss", fontweight="bold")
    ax2.set_ylabel("Learning Rate", fontweight="bold")
    ax1.set_title(f"{title} - Detached LogFreq Autoencoder", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(logfreq_curve_filename)
    plt.close(fig)
    print(f"Detached logfreq training curve saved: {logfreq_curve_filename}")
except Exception as e:
    print(f"Warning: Failed to save logfreq curve plot ({e}).")

# ---------------------------------------------------------------------------
# Section 8: Final Test Evaluation & Leave-One-Out Ablation (norm_phase = 1.0)
# ---------------------------------------------------------------------------
best_model_path = f"{title}_wrapper_best.pt"
print(f"\nLoading composite best weights from '{best_model_path}' for test evaluation...")
raw_model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print("Evaluating test set predictions across all 7 submodules...")
test_metrics = {comp: [] for comp in ALL_COMPONENTS}
collected_preds = {comp: [] for comp in ALL_COMPONENTS}
collected_trues = {comp: [] for comp in ALL_COMPONENTS}

with torch.no_grad():
    for tb_batch in test_loader:
        tb_inputs = [x.to(device=device, dtype=torch.float32) for x in tb_batch]
        tb_recs = model(*tb_inputs[:6], logfreq=tb_inputs[6], return_latent=False)

        for i, comp in enumerate(ALL_COMPONENTS):
            loss_eval = loss_fns[comp](tb_recs[i], tb_inputs[i])
            test_metrics[comp].append(loss_eval.item())
            collected_preds[comp].append(tb_recs[i].cpu().numpy())
            collected_trues[comp].append(tb_inputs[i].cpu().numpy())

for comp in ALL_COMPONENTS:
    collected_preds[comp] = np.concatenate(collected_preds[comp], axis=0)
    collected_trues[comp] = np.concatenate(collected_trues[comp], axis=0)

# Load ground-truth inertial waveforms (2, 2) for the exact test split
print(f"Loading {len(test_indices)} true inertial waveforms from '{DATASET_PATH}'...")
sorted_order = np.argsort(test_indices)
sorted_idx = test_indices[sorted_order]
inv_order = np.argsort(sorted_order)

with h5py.File(DATASET_PATH, "r") as f:
    true_inertial = f["Waveforms"][sorted_idx][inv_order]

print("\nComputing inertial waveform mismatches and leave-one-out error attribution (physical radians)...")
# Unscale predicted and true phases back to physical radians
pred_cooP_rad   = collected_preds["cooP"] * std_cooP + mu_cooP
true_cooP_rad   = collected_trues["cooP"] * std_cooP + mu_cooP
pred_ophase_rad = collected_preds["ophase"] * std_ophase + mu_ophase
true_ophase_rad = collected_trues["ophase"] * std_ophase + mu_ophase

pred_cooA = np.clip(collected_preds["cooA"] * sd_cooA + mu_cooA, 0.0, None)
pred_q    = collected_preds["q"]
true_cooA = collected_trues["cooA"] * sd_cooA + mu_cooA
true_q    = collected_trues["q"]

# Reconstruct phase from predicted log-frequency
pred_cooP_from_logf = to_phase_numpy(collected_preds["logfreq"][:, 0, :], mu_logf, sd_logf)[:, np.newaxis, :]

# 1. Full NN reconstruction (Direct cooP)
mismatches_full = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    norm_phase=1.0,
)

# 2. Substitution: cooP from detached LogFreq module
mismatches_logf = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_from_logf,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    norm_phase=1.0,
)

# 3. Swap ophase -> truth
mismatches_swap_ophase = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_ophase=true_ophase_rad,
    swap_component="ophase",
    norm_phase=1.0,
)

# 4. Swap q -> truth
mismatches_swap_q = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_q=true_q,
    swap_component="q",
    norm_phase=1.0,
)

# 5. Swap cooA -> truth
mismatches_swap_cooA = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_cooA=true_cooA,
    swap_component="cooA",
    norm_phase=1.0,
)

# 6. Swap cooP -> truth
mismatches_swap_cooP = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_cooP=true_cooP_rad,
    swap_component="cooP",
    norm_phase=1.0,
)

# 7. Swap coo (A & P) -> truth
mismatches_swap_coo = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
    pred_ophase=pred_ophase_rad,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_cooA=true_cooA,
    true_cooP=true_cooP_rad,
    swap_component="coo",
    norm_phase=1.0,
)

mismatches_dict = {
    "Full NN (Direct cooP)": mismatches_full,
    "cooP from LogFreq": mismatches_logf,
    "ophase -> truth": mismatches_swap_ophase,
    "q -> truth": mismatches_swap_q,
    "cooA -> truth": mismatches_swap_cooA,
    "cooP -> truth": mismatches_swap_cooP,
    "coo(A+P) -> truth": mismatches_swap_coo,
}

# Save comparative distribution plot
plot_ablation_filename = f"{title}_inertial_mismatch_ablation.png"
plot_inertial_mismatch_ablation(
    mismatches_dict=mismatches_dict,
    run_title=f"{title} (Leave-One-Out)",
    output_path=plot_ablation_filename,
)

# Save test metrics and mismatch arrays
mismatch_npy_filename = f"{title}_inertial_mismatches.npy"
np.save(mismatch_npy_filename, mismatches_dict)

print(f"\n=== Test Set Reconstruction Loss (Normalized Space) ===")
for comp in ALL_COMPONENTS:
    scale_note = f" (physical: {np.mean(test_metrics[comp]) * (std_ophase.mean() if comp=='ophase' else std_cooP.mean()):.4f} rad)" if comp in ("ophase", "cooP") else ""
    print(f"  {comp:8s} | Test SmoothL1: {np.mean(test_metrics[comp]):.6e}{scale_note}")

print(f"\n=== Inertial Waveform Mismatch & Error Attribution (Leave-One-Out) ===")
med_full = float(np.median(mismatches_full))
p95_full = float(np.percentile(mismatches_full, 95))
print(f"  Full NN Prediction:     Median: {med_full:.2e} | 95th: {p95_full:.2e} | Max: {np.max(mismatches_full):.2e}")

for name, mm in mismatches_dict.items():
    if name == "Full NN (Direct cooP)":
        continue
    med_swap = float(np.median(mm))
    delta_med = med_full - med_swap
    pct_drop = (delta_med / med_full * 100.0) if med_full > 0 else 0.0
    print(f"  {name:25s} Median: {med_swap:.2e} (Delta: -{delta_med:.2e} / {pct_drop:+.1f}%)")

print(f"\nSaved artifacts:")
print(f"  - Base Coorbital Model:       {title}_coorbital_best.pt")
print(f"  - Full Wrapper Model:         {title}_wrapper_best.pt")
print(f"  - Checkpoint:                 {title}_checkpoint.pt")
for comp in ALL_COMPONENTS:
    print(f"  - Best {comp:8s} Branch:       {title}_{comp}_best.pt")
print(f"  - Normalization NPZ:          {norm_file}")
print(f"  - History CSV:                {csv_filename}")
print(f"  - History NPY:                {npy_filename}")
print(f"  - Coorbital Curves (2x3):     {plot_curves_filename}")
print(f"  - LogFreq Curve:              {logfreq_curve_filename}")
print(f"  - Inertial Mismatch Ablation: {plot_ablation_filename}")
print(f"  - Inertial Mismatches Data:   {mismatch_npy_filename}")
