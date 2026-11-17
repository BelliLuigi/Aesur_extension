import argparse
import os
import sys
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
import pytorch_optimizer as opt_add

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

from models import CoorbitalAE, init_kaiming_weights
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
# Section 0: Phase Helper Function
# ---------------------------------------------------------------------------
def wave_to_phase(w: np.ndarray) -> np.ndarray:
    """
    Computes unwrapped zero-aligned phase in physical radians.
    w: complex array of shape (N, L)
    returns: phase array of shape (N, L) where phase[:, 0] == 0
    """
    ph = -np.unwrap(np.angle(w), axis=-1)
    return ph - ph[:, :1]


# Component identifiers
COO_COMPONENTS = ["ophase", "q", "chi1", "chi2", "cooA", "cooP"]
ALL_COMPONENTS = COO_COMPONENTS

# ==============================================================================
# Hyperparameters & Configuration
# ==============================================================================
DATASET_PATH     = "datasets/NRSur7dq4v2_dataset.hdf"
SAMPLES          = None         # None to use all available samples (e.g. 102400)
EPOCHS           = 5000         # Total training epochs
BATCH_SIZE       = 128          # Training batch size
VAL_BATCH_SIZE   = 2048         # Validation & test batch size
PRECONDITION_FREQ = 15          # SOAP precondition frequency
BETA             = 0.3          # SmoothL1Loss beta
LR               = 1e-3         # Initial learning rate for SOAP
MIN_LR           = 1e-6         # Minimum learning rate floor
WARMUP_STEPS     = 200          # Linear warmup steps
EARLY_STOP       = 300          # Early stopping patience (epochs, 0 to disable)
LAMBDA_LATENT    = 0.0          # L2 latent regularization weight

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train CoorbitalAE with normalized phases (SOAP optimizer)"
)
parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of samples to load (default: all)")
parser.add_argument("--early_stop", type=int, default=EARLY_STOP, help="Early stopping patience (default: 300)")
parser.add_argument("--lr", type=float, default=LR, help="Initial learning rate (default: 1e-3)")
parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clipping norm threshold (default: 1.0, 0 to disable)")
parser.add_argument("--clamp_coop", type=float, default=0.0, help="Symmetric clamp threshold for normalized cooP (default: 0.0, disabled)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile (default: False)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"CoorbitalAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SAMPLES = args.samples
EARLY_STOP = args.early_stop
LR = args.lr

model_name = "CoorbitalAE"
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

def compute_norm_stats(arr_2d: np.ndarray, train_idx: np.ndarray, floor_frac: float = 0.01):
    """Compute per-timestep mean and std floored at floor_frac * median(std) to avoid float32 noise blow-up."""
    sub = arr_2d[train_idx]
    mu = sub.mean(axis=0)
    raw_std = sub.std(axis=0)
    std_floor = max(floor_frac * float(np.median(raw_std)), 1e-6)
    std = np.maximum(raw_std, std_floor)
    return mu, std

# 1. Normalize ophase using train-set per-timestep mean and floored standard deviation
mu_ophase, std_ophase = compute_norm_stats(ophase_arr[:, 0, :], train_indices)
ophase_arr[:, 0, :] = (ophase_arr[:, 0, :] - mu_ophase) / std_ophase

# 2. Normalize cooP using train-set per-timestep mean and floored standard deviation
mu_cooP, std_cooP = compute_norm_stats(cooP_arr[:, 0, :], train_indices)
cooP_arr[:, 0, :] = (cooP_arr[:, 0, :] - mu_cooP) / std_cooP
if args.clamp_coop > 0:
    n_clamped_coop = int((np.abs(cooP_arr) > args.clamp_coop).sum())
    cooP_arr = np.clip(cooP_arr, -args.clamp_coop, args.clamp_coop)
    print(f"Clamped {n_clamped_coop} cooP outliers to [-{args.clamp_coop}, {args.clamp_coop}] ({100 * n_clamped_coop / cooP_arr.size:.3f}%).")

# 3. Normalize cooA using train-set per-timestep mean and floored standard deviation
mu_cooA, sd_cooA = compute_norm_stats(cooA_arr[:, 0, :], train_indices)
cooA_arr[:, 0, :] = (cooA_arr[:, 0, :] - mu_cooA) / sd_cooA

print(f"Per-timestep normalization applied to ophase, cooP, and cooA.")

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
)
del ophase_arr, q_arr, chi1_arr, chi2_arr, cooA_arr, cooP_arr

j_train = Subset(joint_dataset, train_indices)
j_val   = Subset(joint_dataset, val_indices)
j_test  = Subset(joint_dataset, test_indices)

train_loader = DataLoader(j_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available(), drop_last=True)
val_loader   = DataLoader(j_val, batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(j_test, batch_size=VAL_BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# Model, Loss Functions & Regularization
# ---------------------------------------------------------------------------
raw_model = CoorbitalAE().to(device)
init_kaiming_weights(raw_model)

total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
print(f"Initialized {model_name} with {total_params:,} trainable parameters.")

if args.compile:
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
else:
    print("torch.compile disabled. Running uncompiled eager model.")
    model = raw_model

# Smooth L1 loss per branch (beta=BETA), with time-domain physical weighting
loss_fns = {comp: nn.SmoothL1Loss(beta=BETA, reduction="none") for comp in ALL_COMPONENTS}
lambda_latent = LAMBDA_LATENT

def _make_time_weight(w_np: np.ndarray) -> torch.Tensor:
    w = np.asarray(w_np, dtype=np.float32)
    w_mean = float(w.mean())
    if w_mean > 0.0:
        w = w / w_mean
    return torch.from_numpy(w).view(1, 1, -1).to(device)

loss_weights = {
    "ophase": _make_time_weight(std_ophase * mu_cooA),
    "q": _make_time_weight(mu_cooA),
    "chi1": None,
    "chi2": None,
    "cooA": _make_time_weight(sd_cooA),
    "cooP": _make_time_weight(std_cooP * mu_cooA),
}

def compute_branch_loss(comp: str, pred: torch.Tensor, target: torch.Tensor, latent: torch.Tensor = None) -> torch.Tensor:
    elem_loss = loss_fns[comp](pred, target)
    w = loss_weights[comp]
    loss_c = (elem_loss * w).mean() if w is not None else elem_loss.mean()
    if lambda_latent > 0.0 and latent is not None:
        loss_c = loss_c + lambda_latent * torch.mean(latent ** 2)
    return loss_c

# ---------------------------------------------------------------------------
# Optimizers (SOAP) & Schedulers (Warmup + Cosine Decay) per component
# ---------------------------------------------------------------------------
optimizers = {
    comp: opt_add.SOAP(
        list(getattr(raw_model, f"E_{comp}").parameters()) + list(getattr(raw_model, f"D_{comp}").parameters()),
        lr=LR,
        betas=(0.95, 0.95),
        shampoo_beta=0.99,
        weight_decay=0.0,
        precondition_frequency=PRECONDITION_FREQ,
        merge_dims=False,
        eps=1e-8,
    )
    for comp in ALL_COMPONENTS
}

total_steps = EPOCHS * max(len(train_loader), 1)
warmup_steps = min(WARMUP_STEPS, max(1, total_steps // 2))
decay_steps = max(total_steps - warmup_steps, 1)

def make_warmup_cosine_scheduler(optimizer):
    warmup_sched = lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    decay_sched = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=MIN_LR
    )
    return lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps]
    )

schedulers = {comp: make_warmup_cosine_scheduler(optimizers[comp]) for comp in ALL_COMPONENTS}
print(f"Schedule: {warmup_steps} warmup steps ({1e-6:.1e} -> {LR:.1e}) + {decay_steps} cosine steps -> {MIN_LR:.1e} (Total: {total_steps} steps across {EPOCHS} epochs)")

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
        f"Early stopping enabled for all 6 branches: patience={EARLY_STOP}, "
        f"floor_patience=50 at min_lr={MIN_LR:.1e} (gated to final 15%: epoch >= {int(0.85 * EPOCHS)})"
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
best_coorbital_state = {k: v.clone() for k, v in raw_model.state_dict().items()}

print(f"\nStarting {model_name} training for {EPOCHS} epochs...\n")

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss_sums = {comp: 0.0 for comp in ALL_COMPONENTS}

    for batch in train_loader:
        b_inputs = [x.to(device=device, dtype=torch.float32, non_blocking=True) for x in batch]
        # b_inputs order: ophase, q, chi1, chi2, cooA, cooP

        for comp in ALL_COMPONENTS:
            if not stopped[comp]:
                optimizers[comp].zero_grad()

        recs, latents = model(*b_inputs, return_latent=True)

        total_active_loss = torch.tensor(0.0, device=device)
        for i, comp in enumerate(ALL_COMPONENTS):
            loss_c = compute_branch_loss(comp, recs[i], b_inputs[i], latents[i])

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
                    schedulers[comp].step()

    n_train_batches = max(len(train_loader), 1)
    avg_train_losses = {comp: train_loss_sums[comp] / n_train_batches for comp in ALL_COMPONENTS}

    # Validation
    model.eval()
    val_loss_sums = {comp: 0.0 for comp in ALL_COMPONENTS}
    with torch.no_grad():
        for vbatch in val_loader:
            vb_inputs = [x.to(device=device, dtype=torch.float32, non_blocking=True) for x in vbatch]
            vrecs, vz = model(*vb_inputs, return_latent=True)
            for i, comp in enumerate(ALL_COMPONENTS):
                vloss_c = compute_branch_loss(comp, vrecs[i], vb_inputs[i], vz[i])
                val_loss_sums[comp] += vloss_c.item()

    n_val_batches = max(len(val_loader), 1)
    avg_val_losses = {comp: val_loss_sums[comp] / n_val_batches for comp in ALL_COMPONENTS}

    # Read current LRs (stepped per batch)
    current_lrs = {comp: schedulers[comp].get_last_lr()[0] for comp in ALL_COMPONENTS}

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

            # Update composite optimal state
            any_coo_improved = True
            for k, v in raw_model.state_dict().items():
                if k.startswith(f"E_{comp}.") or k.startswith(f"D_{comp}."):
                    best_coorbital_state[k] = v.clone()

    if any_coo_improved:
        torch.save(best_coorbital_state, f"{title}_coorbital_best.pt")

    # Save periodic full training checkpoint
    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "best_coorbital_state_dict": best_coorbital_state,
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

    # Check Early Stopping (gated to final 15% of training)
    if EARLY_STOP > 0 and epoch >= int(0.85 * EPOCHS):
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

# ---------------------------------------------------------------------------
# Section 8: Final Test Evaluation & Leave-One-Out Ablation (norm_phase = 1.0)
# ---------------------------------------------------------------------------
best_model_path = f"{title}_coorbital_best.pt"
print(f"\nLoading composite best weights from '{best_model_path}' for test evaluation...")
raw_model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print("Evaluating test set predictions across all 6 submodules...")
test_metrics = {comp: [] for comp in ALL_COMPONENTS}
collected_preds = {comp: [] for comp in ALL_COMPONENTS}
collected_trues = {comp: [] for comp in ALL_COMPONENTS}

with torch.no_grad():
    for tb_batch in test_loader:
        tb_inputs = [x.to(device=device, dtype=torch.float32) for x in tb_batch]
        tb_recs = model(*tb_inputs, return_latent=False)

        for i, comp in enumerate(ALL_COMPONENTS):
            loss_eval = compute_branch_loss(comp, tb_recs[i], tb_inputs[i])
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

# 1. Full NN reconstruction
mismatches_full = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP_rad,
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
    "Full NN": mismatches_full,
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
    if name == "Full NN":
        continue
    med_swap = float(np.median(mm))
    delta_med = med_full - med_swap
    pct_drop = (delta_med / med_full * 100.0) if med_full > 0 else 0.0
    print(f"  {name:25s} Median: {med_swap:.2e} (Delta: -{delta_med:.2e} / {pct_drop:+.1f}%)")

print(f"\nSaved artifacts:")
print(f"  - Best Coorbital Model:       {title}_coorbital_best.pt")
print(f"  - Checkpoint:                 {title}_checkpoint.pt")
for comp in ALL_COMPONENTS:
    print(f"  - Best {comp:8s} Branch:       {title}_{comp}_best.pt")
print(f"  - Normalization NPZ:          {norm_file}")
print(f"  - History CSV:                {csv_filename}")
print(f"  - History NPY:                {npy_filename}")
print(f"  - Coorbital Curves (2x3):     {plot_curves_filename}")
print(f"  - Inertial Mismatch Ablation: {plot_ablation_filename}")
print(f"  - Inertial Mismatches Data:   {mismatch_npy_filename}")
