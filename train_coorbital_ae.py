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
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_optimizer as opt_add

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

from models import CoorbitalAE
from utils import (
    get_phase,
    plot_coorbital_training_curves,
    TasosOverlapTorch,
    plot_mismatch_distribution,
    EarlyStopping,
    reconstruct_inertial_waveform,
    compute_inertial_mismatches,
    plot_inertial_mismatch_ablation,
)

COMPONENT_NAMES = ["ophase", "q", "chi1", "chi2", "cooA", "cooP"]

# ==============================================================================
# Hyperparameters & Configuration (Customizable)
# ==============================================================================
DATASET_PATH     = "datasets/NRSur7dq4v2_dataset.hdf"
SAMPLES          = None         # None to use all available samples (e.g. 102400)
EPOCHS           = 5000         # Total training epochs
BATCH_SIZE       = 128          # Training batch size
VAL_BATCH_SIZE   = 2048         # Validation & test batch size
LR               = 5e-3         # Initial learning rate for SOAP
MIN_LR           = 5e-7         # Minimum learning rate floor
LR_PATIENCE      = 100          # ReduceLROnPlateau patience (epochs)
EARLY_STOP       = 300          # Early stopping patience (epochs, 0 to disable)
LAMBDA_LATENT    = 1e-6         # L2 latent regularization weight
NORM_PHASE       = 100.0        # Scaling factor for phase values (ophase and cooP)

# ---------------------------------------------------------------------------
# CLI Arguments (Minimal Execution Controls)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train CoorbitalAE (6 modular branches: ophase, q, chi1, chi2, cooA, cooP) with SOAP optimizer"
)
parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"CoorbitalAE_soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_name = "CoorbitalAE"
print(f"=== Starting Run: {title} | Model: {model_name} | Optimizer: SOAP | Scheduler: PLATEAU | Epochs: {EPOCHS} ===")

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

print(f"Loading coorbital kinematic and dynamic components from '{filepath}'...")
with h5py.File(filepath, "r") as f:
    total_avail = f["Waveforms_coorb"].shape[0]
    n_samples = min(SAMPLES, total_avail) if (SAMPLES and SAMPLES > 0) else total_avail
    sample_len = f["Waveforms_coorb"].shape[1]

    ophase_arr = np.empty((n_samples, 1, sample_len), dtype=np.float32)
    q_arr = np.empty((n_samples, 4, sample_len), dtype=np.float32)
    chi1_arr = np.empty((n_samples, 3, sample_len), dtype=np.float32)
    chi2_arr = np.empty((n_samples, 3, sample_len), dtype=np.float32)
    cooA_arr = np.empty((n_samples, 1, sample_len), dtype=np.float32)
    cooP_arr = np.empty((n_samples, 1, sample_len), dtype=np.float32)

    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        cur_n = end - start

        # 1. Orbital Phase (N, 1, 2048) scaled by 1/NORM_PHASE to match O(1) range
        ophase_arr[start:end, 0, :] = f["Dynamics/orbphase"][start:end].astype(np.float32) / NORM_PHASE

        # 2. Coprecessing Quaternions (N, 4, 2048)
        q_arr[start:end] = f["Dynamics/q_copr"][start:end].astype(np.float32)

        # 3. Dynamic spins chiA (N, 2048, 3) -> transpose to (N, 3, 2048)
        chi1_arr[start:end] = np.transpose(f["Dynamics/chiA"][start:end].astype(np.float32), (0, 2, 1))

        # 4. Dynamic spins chiB (N, 2048, 3) -> transpose to (N, 3, 2048)
        chi2_arr[start:end] = np.transpose(f["Dynamics/chiB"][start:end].astype(np.float32), (0, 2, 1))

        # 5 & 6. Co-orbital Waveform (2,2): Amplitude and Unwrapped Phase (N, 1, 2048)
        wfs_coorb = f["Waveforms_coorb"][start:end]
        cooA_arr[start:end, 0, :] = np.abs(wfs_coorb).astype(np.float32)
        cooP_arr[start:end, 0, :] = get_phase(wfs_coorb).astype(np.float32)

        if (end % 20000 == 0) or (end == n_samples):
            print(f"  Loaded {end}/{n_samples} samples ({end / n_samples * 100:.1f}%)")

ophase_tensor = torch.from_numpy(ophase_arr)
q_tensor = torch.from_numpy(q_arr)
chi1_tensor = torch.from_numpy(chi1_arr)
chi2_tensor = torch.from_numpy(chi2_arr)
cooA_tensor = torch.from_numpy(cooA_arr)
cooP_tensor = torch.from_numpy(cooP_arr)

del ophase_arr, q_arr, chi1_arr, chi2_arr, cooA_arr, cooP_arr

joint_dataset = TensorDataset(
    ophase_tensor,
    q_tensor,
    chi1_tensor,
    chi2_tensor,
    cooA_tensor,
    cooP_tensor,
)

# 80% train / 10% val / 10% test split
j_train, j_val, j_test = random_split(
    joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    j_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
val_loader = DataLoader(
    j_val,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    j_test,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
)

print(f"Dataset split: Train={len(j_train)} | Val={len(j_val)} | Test={len(j_test)}")

# ---------------------------------------------------------------------------
# Model, Loss Functions & Regularization
# ---------------------------------------------------------------------------
raw_model = CoorbitalAE().to(device)

total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
print(f"Initialized CoorbitalAE with {total_params:,} trainable parameters across 12 submodules.")

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
loss_fns = {comp: nn.SmoothL1Loss(beta=1e-3) for comp in COMPONENT_NAMES}
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
        precondition_frequency=25,
        merge_dims=False,
        eps=0.0001,
    )
    for comp in COMPONENT_NAMES
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
    for comp in COMPONENT_NAMES
}

# ---------------------------------------------------------------------------
# Early Stopping Initialization
# ---------------------------------------------------------------------------
stopped = {comp: False for comp in COMPONENT_NAMES}
best_vloss = {comp: float("inf") for comp in COMPONENT_NAMES}

if EARLY_STOP > 0:
    early_stoppers = {
        comp: EarlyStopping(
            patience=EARLY_STOP,
            threshold=0.001,
            mode="rel",
            min_lr=MIN_LR,
            patience_at_min_lr=50,
        )
        for comp in COMPONENT_NAMES
    }
    print(
        f"Early stopping enabled for all 6 branches: patience={EARLY_STOP}, "
        f"floor_patience=50 at min_lr={MIN_LR:.1e} (rel threshold=0.1%)"
    )
else:
    early_stoppers = {}
    print("Early stopping disabled.")

history = {"epoch": []}
for comp in COMPONENT_NAMES:
    history[f"train_loss_{comp}"] = []
    history[f"val_loss_{comp}"] = []
    history[f"lr_{comp}"] = []

# Keep optimal weights composed across the best epoch of each branch
best_full_state = {k: v.clone() for k, v in raw_model.state_dict().items()}

print(f"\nStarting CoorbitalAE training for {EPOCHS} epochs...\n")

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss_sums = {comp: 0.0 for comp in COMPONENT_NAMES}

    for batch in train_loader:
        b_inputs = [x.to(device=device, dtype=torch.float32, non_blocking=True) for x in batch]
        # b_inputs order: ophase, q, chi1, chi2, cooA, cooP

        for comp in COMPONENT_NAMES:
            if not stopped[comp]:
                optimizers[comp].zero_grad()

        recs, latents = model(*b_inputs, return_latent=True)

        total_active_loss = torch.tensor(0.0, device=device)
        for i, comp in enumerate(COMPONENT_NAMES):
            loss_c = loss_fns[comp](recs[i], b_inputs[i])
            if lambda_latent > 0.0:
                loss_c = loss_c + lambda_latent * torch.mean(latents[i] ** 2)

            train_loss_sums[comp] += loss_c.item()

            if not stopped[comp]:
                total_active_loss = total_active_loss + loss_c

        if total_active_loss.requires_grad:
            total_active_loss.backward()
            for comp in COMPONENT_NAMES:
                if not stopped[comp]:
                    optimizers[comp].step()

    n_train_batches = max(len(train_loader), 1)
    avg_train_losses = {comp: train_loss_sums[comp] / n_train_batches for comp in COMPONENT_NAMES}

    # Validation
    model.eval()
    val_loss_sums = {comp: 0.0 for comp in COMPONENT_NAMES}
    with torch.no_grad():
        for vbatch in val_loader:
            vb_inputs = [x.to(device=device, dtype=torch.float32, non_blocking=True) for x in vbatch]
            vrecs, vz = model(*vb_inputs, return_latent=True)
            for i, comp in enumerate(COMPONENT_NAMES):
                vloss_c = loss_fns[comp](vrecs[i], vb_inputs[i])
                if lambda_latent > 0.0:
                    vloss_c = vloss_c + lambda_latent * torch.mean(vz[i] ** 2)
                val_loss_sums[comp] += vloss_c.item()

    n_val_batches = max(len(val_loader), 1)
    avg_val_losses = {comp: val_loss_sums[comp] / n_val_batches for comp in COMPONENT_NAMES}

    # Step LR Schedulers
    current_lrs = {}
    for comp in COMPONENT_NAMES:
        schedulers[comp].step(avg_val_losses[comp])
        current_lrs[comp] = schedulers[comp].get_last_lr()[0]

    # Record History
    history["epoch"].append(epoch + 1)
    for comp in COMPONENT_NAMES:
        history[f"train_loss_{comp}"].append(avg_train_losses[comp])
        history[f"val_loss_{comp}"].append(avg_val_losses[comp])
        history[f"lr_{comp}"].append(current_lrs[comp])

    # Check and save best per-branch models and update composite best full model
    any_improved = False
    for comp in COMPONENT_NAMES:
        if avg_val_losses[comp] < best_vloss[comp]:
            best_vloss[comp] = avg_val_losses[comp]
            any_improved = True

            # Save standalone sub-autoencoder state for this component
            sub_state = {
                f"E_{comp}": getattr(raw_model, f"E_{comp}").state_dict(),
                f"D_{comp}": getattr(raw_model, f"D_{comp}").state_dict(),
            }
            torch.save(sub_state, f"{title}_{comp}_best.pt")

            # Update composite optimal full state dict
            for k, v in raw_model.state_dict().items():
                if k.startswith(f"E_{comp}.") or k.startswith(f"D_{comp}."):
                    best_full_state[k] = v.clone()

    if any_improved:
        torch.save(best_full_state, f"{title}_coorbital_best.pt")

    # Save periodic full training checkpoint (including optimizer & scheduler states)
    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "best_full_state_dict": best_full_state,
                "optimizers": {comp: optimizers[comp].state_dict() for comp in COMPONENT_NAMES},
                "schedulers": {comp: schedulers[comp].state_dict() for comp in COMPONENT_NAMES},
                "best_vloss": best_vloss,
            },
            f"{title}_coorbital_checkpoint.pt",
        )

    # Logging
    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == EPOCHS:
        print(f"--- Epoch [{epoch + 1}/{EPOCHS}] ---")
        for comp in COMPONENT_NAMES:
            status = "STOPPED" if stopped[comp] else f"LR: {current_lrs[comp]:.2e}"
            print(
                f"  {comp:8s} | Train Loss: {avg_train_losses[comp]:.6f} | "
                f"Val Loss: {avg_val_losses[comp]:.6f} | Best: {best_vloss[comp]:.6f} | {status}"
            )

    # Check Early Stopping
    if EARLY_STOP > 0:
        for comp in COMPONENT_NAMES:
            if not stopped[comp]:
                stopped[comp] = early_stoppers[comp].step(avg_val_losses[comp], current_lrs[comp])

        if all(stopped.values()):
            print(f"\n=== [Early Stopping Triggered for all branches at Epoch {epoch + 1}] ===")
            for comp in COMPONENT_NAMES:
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
# Generate Professional Training Curve Plot (Unified 6 Subplots)
# ---------------------------------------------------------------------------
print("\nGenerating unified 6-subplot training curves...")
plot_curves_filename = f"{title}_training_curves.png"
plot_coorbital_training_curves(
    history=history,
    component_names=COMPONENT_NAMES,
    run_title=title,
    output_path=plot_curves_filename,
)

print(f"\nTraining completed successfully for run '{title}'.")

# ---------------------------------------------------------------------------
# Evaluate Best Saved Models on Test Set & Inertial Waveform Mismatches (with Ablation)
# ---------------------------------------------------------------------------
best_model_path = f"{title}_coorbital_best.pt"
print(f"\nLoading composite best weights from '{best_model_path}' for test evaluation...")
raw_model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print("Evaluating test set predictions across all 6 submodules...")
test_metrics = {comp: [] for comp in COMPONENT_NAMES}

collected_preds = {comp: [] for comp in COMPONENT_NAMES}
collected_trues = {comp: [] for comp in COMPONENT_NAMES}

with torch.no_grad():
    for tb_batch in test_loader:
        tb_inputs = [x.to(device=device, dtype=torch.float32) for x in tb_batch]
        tb_recs = model(*tb_inputs, return_latent=False)

        for i, comp in enumerate(COMPONENT_NAMES):
            loss_eval = loss_fns[comp](tb_recs[i], tb_inputs[i])
            test_metrics[comp].append(loss_eval.item())
            collected_preds[comp].append(tb_recs[i].cpu().numpy())
            collected_trues[comp].append(tb_inputs[i].cpu().numpy())

for comp in COMPONENT_NAMES:
    collected_preds[comp] = np.concatenate(collected_preds[comp], axis=0)
    collected_trues[comp] = np.concatenate(collected_trues[comp], axis=0)

# Load ground-truth inertial waveforms (2, 2) from dataset for the exact test split
print(f"Loading {len(j_test)} true inertial waveforms from '{DATASET_PATH}'...")
test_indices = np.array(j_test.indices)
sorted_order = np.argsort(test_indices)
sorted_idx = test_indices[sorted_order]
inv_order = np.argsort(sorted_order)

with h5py.File(DATASET_PATH, "r") as f:
    true_inertial = f["Waveforms"][sorted_idx][inv_order]

print("\nComputing inertial waveform mismatches and leave-one-out error attribution...")
pred_cooA = collected_preds["cooA"]
pred_cooP = collected_preds["cooP"]
pred_ophase = collected_preds["ophase"]
pred_q = collected_preds["q"]

true_cooA = collected_trues["cooA"]
true_cooP = collected_trues["cooP"]
true_ophase = collected_trues["ophase"]
true_q = collected_trues["q"]

# 1. Full model reconstruction (all predicted)
mismatches_full = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP,
    pred_ophase=pred_ophase,
    pred_q=pred_q,
    true_inertial=true_inertial,
    norm_phase=NORM_PHASE,
)

# 2. Swap ophase -> truth (isolate orbital phase contribution)
mismatches_swap_ophase = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP,
    pred_ophase=pred_ophase,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_ophase=true_ophase,
    swap_component="ophase",
    norm_phase=NORM_PHASE,
)

# 3. Swap q -> truth (isolate quaternion contribution)
mismatches_swap_q = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP,
    pred_ophase=pred_ophase,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_q=true_q,
    swap_component="q",
    norm_phase=NORM_PHASE,
)

# 4. Swap coo (A & P) -> truth (isolate co-orbital waveform mode contribution)
mismatches_swap_coo = compute_inertial_mismatches(
    pred_cooA=pred_cooA,
    pred_cooP=pred_cooP,
    pred_ophase=pred_ophase,
    pred_q=pred_q,
    true_inertial=true_inertial,
    true_cooA=true_cooA,
    true_cooP=true_cooP,
    swap_component="coo",
    norm_phase=NORM_PHASE,
)

mismatches_dict = {
    "Full NN": mismatches_full,
    "ophase -> truth": mismatches_swap_ophase,
    "q -> truth": mismatches_swap_q,
    "coo -> truth": mismatches_swap_coo,
}

# Save comparative distribution plot (Fig. 8 style)
plot_ablation_filename = f"{title}_inertial_mismatch_ablation.png"
plot_inertial_mismatch_ablation(
    mismatches_dict=mismatches_dict,
    run_title=title,
    output_path=plot_ablation_filename,
)

# Save test metrics and mismatch arrays
mismatch_npy_filename = f"{title}_inertial_mismatches.npy"
np.save(mismatch_npy_filename, mismatches_dict)

print(f"\n=== Test Set Reconstruction Loss (Normalized) ===")
for comp in COMPONENT_NAMES:
    scale_note = f" (physical: {np.mean(test_metrics[comp]) * NORM_PHASE:.4f} rad)" if comp in ("ophase", "cooP") else ""
    print(f"  {comp:8s} | Test SmoothL1: {np.mean(test_metrics[comp]):.6e}{scale_note}")

print(f"\n=== Inertial Waveform Mismatch & Error Attribution (Leave-One-Out) ===")
med_full = float(np.median(mismatches_full))
p95_full = float(np.percentile(mismatches_full, 95))
print(f"  Full NN Prediction:     Median: {med_full:.2e} | 95th: {p95_full:.2e} | Max: {np.max(mismatches_full):.2e}")

for name, mm in [("ophase -> truth", mismatches_swap_ophase), ("q -> truth", mismatches_swap_q), ("coo -> truth", mismatches_swap_coo)]:
    med_swap = float(np.median(mm))
    delta_med = med_full - med_swap
    pct_drop = (delta_med / med_full * 100.0) if med_full > 0 else 0.0
    print(f"  {name:22s} Median: {med_swap:.2e} (Delta: -{delta_med:.2e} / {pct_drop:+.1f}%)")

print(f"\nSaved artifacts:")
print(f"  - Full Best Coorbital Model:  {title}_coorbital_best.pt")
print(f"  - Full Checkpoint:            {title}_coorbital_checkpoint.pt")
for comp in COMPONENT_NAMES:
    print(f"  - Best {comp:8s} Branch:       {title}_{comp}_best.pt")
print(f"  - History CSV:                {csv_filename}")
print(f"  - History NPY:                {npy_filename}")
print(f"  - Training Curves (6 in 1):   {plot_curves_filename}")
print(f"  - Inertial Mismatch Ablation: {plot_ablation_filename}")
print(f"  - Inertial Mismatches Data:   {mismatch_npy_filename}")
