import argparse
import os
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, TensorDataset
import pytorch_optimizer as opt_add
import sxs
from pycbc import conversions

from models import Autoencoder_CNN_NOBN, SimplestMapper
from utils import (
    get_phase,
    get_NR_data,
    farthest_point_sampling,
    plot_training_curves,
    plot_mismatch_distribution,
    EarlyStopping,
    TasosOverlapTorch,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(device)
    print(f"Allocated GPU: {prop.name} | Compute Capability: {prop.major}.{prop.minor} | VRAM: {prop.total_memory / (1024**3):.1f} GB")
else:
    print("Using CPU device.")

# ---------------------------------------------------------------------------
# Hyperparameters & Paths
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
K_FOLDS = 5
CHECKPOINT_AMPL_PATH = "model_weights/soap2110_amplitude_best.pt"
CHECKPOINT_PHASE_PATH = "model_weights/soap2110_phase_best.pt"

# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Finetune FullModels with K-Fold CV on SXS (NRHybSur3dq8)")
parser.add_argument("--title", type=str, default=None, help="Title / prefix for this run")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs per fold (default: 500)")
parser.add_argument("--early_stop", type=int, default=100, help="Early stopping patience in epochs (default: 100, 0 to disable)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"Finetune_SOAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"=== Starting 5-Fold Finetuning Run: {title} | Epochs/fold: {args.epochs} | Scheduler: ReduceLROnPlateau ===")

# ---------------------------------------------------------------------------
# Architecture: FullModel with Unfrozen Decoder
# ---------------------------------------------------------------------------
class FullModel(nn.Module):
    def __init__(self, mapper: nn.Module, decoder: nn.Module):
        super().__init__()
        self.mapper = mapper
        self.decoder = decoder

    def forward(self, x):
        z = self.mapper(x)
        out = self.decoder(z)
        if out.ndim == 3 and out.size(1) == 1:
            out = out.squeeze(1)
        return out


def build_full_models():
    """Initializes fresh mappers and reloads pretrained decoder weights for each fold."""
    model_amplitude = SimplestMapper().to(device)
    model_phase = SimplestMapper().to(device)

    ae_amplitude = Autoencoder_CNN_NOBN().to(device)
    ae_phase = Autoencoder_CNN_NOBN().to(device)

    if os.path.exists(CHECKPOINT_AMPL_PATH):
        ae_amplitude.load_state_dict(torch.load(CHECKPOINT_AMPL_PATH, map_location=device), strict=False)
    else:
        print(f"[Warning] Pretrained amplitude checkpoint '{CHECKPOINT_AMPL_PATH}' not found. Initializing decoder randomly.")

    if os.path.exists(CHECKPOINT_PHASE_PATH):
        ae_phase.load_state_dict(torch.load(CHECKPOINT_PHASE_PATH, map_location=device), strict=False)
    else:
        print(f"[Warning] Pretrained phase checkpoint '{CHECKPOINT_PHASE_PATH}' not found. Initializing decoder randomly.")

    filter_amplitude = ae_amplitude.decoder
    filter_phase = ae_phase.decoder
    filter_amplitude.requires_grad_(True)
    filter_phase.requires_grad_(True)

    full_model_ampl = FullModel(model_amplitude, filter_amplitude).to(device)
    full_model_phase = FullModel(model_phase, filter_phase).to(device)
    return full_model_ampl, full_model_phase


# ---------------------------------------------------------------------------
# Load SXS Data
# ---------------------------------------------------------------------------
SXS_path = (
    "datasets/sxs_waveforms.h5"
    if os.path.exists("datasets/sxs_waveforms.h5")
    else "sxs_waveforms.h5"
)

with h5py.File(SXS_path, "r") as f:
    names = list(f.keys())
    wavs = [get_NR_data(f[str(name)], "waveform") for name in names]
    params = [get_NR_data(f[str(name)], "metadata") for name in names]
    mass = 100 * sxs.m_sun_in_seconds
    R = 100 * 1e6 * sxs.parsec_in_meters

    good_wavs = np.array([(len(x) == 2048) for x in wavs])
    wavs = np.array(wavs)[good_wavs]
    names = np.array(names)[good_wavs]
    params = np.array(params)[good_wavs]

    mask = params[:, 0] <= 8
    wavs = wavs[mask]
    params = np.array(params)[mask]
    names = np.array(names)[mask]

    [waves_sxs, params_sxs, names_sxs] = np.stack(wavs), np.stack(params), np.array(names)
    filt = (
        (abs(params_sxs[:, -2]) < 1e-2)
        * (abs(params_sxs[:, 1]) <= 0.8)
        * (abs(params_sxs[:, 1]) <= 0.8)
        * (abs(waves_sxs[:, :]).sum(axis=-1) < 1000)
        * (abs(params_sxs[:, -1]) < 1e-2)
        * (abs(waves_sxs[:, -1]) < 0.5)
    )
    waves_sxs = waves_sxs[filt]
    params_sxs = params_sxs[filt]
    names_sxs = names_sxs[filt]

params_sxs = params_sxs[:, :3]
params_sxs[:, 0] = 1 / params_sxs[:, 0]

print(f"Loaded {len(names_sxs)} SXS waveforms: waves {waves_sxs.shape}, params {params_sxs.shape}")

params_tensor = torch.from_numpy(params_sxs).float()
ampl_tensor = torch.from_numpy(np.abs(waves_sxs)).float()
phase_tensor = torch.from_numpy(get_phase(waves_sxs)).float()
joint_dataset = TensorDataset(params_tensor, ampl_tensor, phase_tensor)

# ---------------------------------------------------------------------------
# 1. Held-Out Uniform Test Set (10% ~ 37 samples) via FPS
# ---------------------------------------------------------------------------
n_test = int(0.10 * len(joint_dataset))
test_idx = farthest_point_sampling(params_tensor, n_test)

mask = torch.ones(len(joint_dataset), dtype=torch.bool)
mask[test_idx] = False
train_val_idx = torch.arange(len(joint_dataset))[mask]

test_loader = DataLoader(Subset(joint_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
print(f"Test set (held-out): {len(test_idx)} samples | Cross-validation pool: {len(train_val_idx)} samples")

# ---------------------------------------------------------------------------
# 2. K-Fold Cross-Validation Loop
# ---------------------------------------------------------------------------
k = K_FOLDS
perm = train_val_idx[torch.randperm(len(train_val_idx), generator=torch.Generator().manual_seed(42))]
folds = torch.chunk(perm, chunks=k)

cv_fold_summaries = []
best_overall_mismatch = float("inf")
best_overall_fold = 0

for fold in range(k):
    fold_num = fold + 1
    fold_title = f"{title}_fold{fold_num}"
    print(f"\n==================================================")
    print(f"  Starting Fold {fold_num}/{k} | Run: {fold_title}")
    print(f"==================================================")

    val_idx = folds[fold]
    train_idx = torch.cat([folds[i] for i in range(k) if i != fold])

    train_loader = DataLoader(Subset(joint_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(joint_dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Fold {fold_num}: Train {len(train_idx)} samples | Val {len(val_idx)} samples")

    # Reset models and reload base weights for this fold
    full_model_ampl, full_model_phase = build_full_models()

    loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3)
    loss_fn_phase = nn.SmoothL1Loss(beta=1e-3)

    opt_fn_amplitude = opt_add.SOAP(
        full_model_ampl.parameters(), lr=3e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
    opt_fn_phase = opt_add.SOAP(
        full_model_phase.parameters(), lr=3e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )

    scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
        opt_fn_amplitude, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=25
    )
    scheduler_phase = lr_scheduler.ReduceLROnPlateau(
        opt_fn_phase, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=25
    )

    if args.early_stop > 0:
        early_stop_amplitude = EarlyStopping(
            patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-6, patience_at_min_lr=50
        )
        early_stop_phase = EarlyStopping(
            patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-6, patience_at_min_lr=50
        )

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

    # Epoch training loop
    for epoch in range(args.epochs):
        # 1. Train Step
        full_model_ampl.train(True)
        full_model_phase.train(True)
        total_loss_amplitudeT = 0.0
        total_loss_phaseT = 0.0

        for b_params, b_ampl, b_phase in train_loader:
            b_params = b_params.float().to(device)
            b_ampl = b_ampl.float().to(device)
            b_phase = b_phase.float().to(device)

            # Amplitude step
            opt_fn_amplitude.zero_grad()
            pred_wf_ampl = full_model_ampl(b_params)
            loss_amp = loss_fn_amplitude(pred_wf_ampl, b_ampl)
            loss_amp.backward()
            opt_fn_amplitude.step()
            total_loss_amplitudeT += loss_amp.item()

            # Phase step
            opt_fn_phase.zero_grad()
            pred_wf_phase = full_model_phase(b_params)
            loss_phase = loss_fn_phase(pred_wf_phase, b_phase)
            loss_phase.backward()
            opt_fn_phase.step()
            total_loss_phaseT += loss_phase.item()

        total_loss_amplitudeT /= max(len(train_loader), 1)
        total_loss_phaseT /= max(len(train_loader), 1)

        # 2. Validation Step
        full_model_ampl.eval()
        full_model_phase.eval()
        total_loss_amplitudeV = 0.0
        total_loss_phaseV = 0.0

        with torch.no_grad():
            for b_params, b_ampl, b_phase in val_loader:
                b_params = b_params.float().to(device)
                b_ampl = b_ampl.float().to(device)
                b_phase = b_phase.float().to(device)

                pred_wf_ampl = full_model_ampl(b_params)
                loss_amp = loss_fn_amplitude(pred_wf_ampl, b_ampl)
                total_loss_amplitudeV += loss_amp.item()

                pred_wf_phase = full_model_phase(b_params)
                loss_phase = loss_fn_phase(pred_wf_phase, b_phase)
                total_loss_phaseV += loss_phase.item()

        total_loss_amplitudeV /= max(len(val_loader), 1)
        total_loss_phaseV /= max(len(val_loader), 1)

        # 3. LR Scheduler Step
        scheduler_amplitude.step(total_loss_amplitudeV)
        scheduler_phase.step(total_loss_phaseV)

        current_lr_ampl = scheduler_amplitude.get_last_lr()[0]
        current_lr_ph = scheduler_phase.get_last_lr()[0]

        history["epoch"].append(epoch + 1)
        history["train_loss_amplitude"].append(total_loss_amplitudeT)
        history["val_loss_amplitude"].append(total_loss_amplitudeV)
        history["lr_amplitude"].append(current_lr_ampl)
        history["train_loss_phase"].append(total_loss_phaseT)
        history["val_loss_phase"].append(total_loss_phaseV)
        history["lr_phase"].append(current_lr_ph)

        # 4. Save Best Model Checkpoints for this fold
        if total_loss_amplitudeV < best_vloss_amplitude:
            best_vloss_amplitude = total_loss_amplitudeV
            torch.save(full_model_ampl.state_dict(), f"{fold_title}_amplitude_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": full_model_ampl.state_dict(),
                    "optimizer_state_dict": opt_fn_amplitude.state_dict(),
                    "scheduler_state_dict": scheduler_amplitude.state_dict(),
                    "best_vloss": best_vloss_amplitude,
                },
                f"{fold_title}_amplitude_checkpoint.pt",
            )

        if total_loss_phaseV < best_vloss_phase:
            best_vloss_phase = total_loss_phaseV
            torch.save(full_model_phase.state_dict(), f"{fold_title}_phase_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": full_model_phase.state_dict(),
                    "optimizer_state_dict": opt_fn_phase.state_dict(),
                    "scheduler_state_dict": scheduler_phase.state_dict(),
                    "best_vloss": best_vloss_phase,
                },
                f"{fold_title}_phase_checkpoint.pt",
            )

        # 5. Early Stopping Check
        if args.early_stop > 0:
            stop_a = early_stop_amplitude.step(total_loss_amplitudeV, current_lr_ampl)
            stop_p = early_stop_phase.step(total_loss_phaseV, current_lr_ph)
            if stop_a and stop_p:
                print(f"[Fold {fold_num}] Early Stopping at Epoch {epoch + 1} (Ampl: {early_stop_amplitude.stop_reason}, Phase: {early_stop_phase.stop_reason})")
                break

    # Save fold history CSV and training curves
    df_history = pd.DataFrame(history)
    csv_filename = f"{fold_title}_metrics.csv"
    df_history.to_csv(csv_filename, index=False)

    plot_training_curves(
        epochs=history["epoch"],
        train_losses=history["train_loss_amplitude"],
        val_losses=history["val_loss_amplitude"],
        lrs=history["lr_amplitude"],
        run_title=fold_title,
        metric_name="Amplitude",
        output_path=f"{fold_title}_amplitude_plot.png",
    )
    plot_training_curves(
        epochs=history["epoch"],
        train_losses=history["train_loss_phase"],
        val_losses=history["val_loss_phase"],
        lrs=history["lr_phase"],
        run_title=fold_title,
        metric_name="Phase",
        output_path=f"{fold_title}_phase_plot.png",
    )

    cv_fold_summaries.append({
        "fold": fold_num,
        "best_val_loss_ampl": best_vloss_amplitude,
        "best_val_loss_phase": best_vloss_phase,
        "epochs_trained": len(history["epoch"]),
    })
    print(f"[Fold {fold_num} Done] Best Val Ampl Loss: {best_vloss_amplitude:.6e} | Best Val Phase Loss: {best_vloss_phase:.6e}")

# ---------------------------------------------------------------------------
# Cross-Validation Summary & Final Test Set Evaluation
# ---------------------------------------------------------------------------
df_cv = pd.DataFrame(cv_fold_summaries)
df_cv.to_csv(f"{title}_cv_summary.csv", index=False)
print("\n==================================================")
print(f"=== Cross-Validation Results across {k} Folds ===")
print("==================================================")
print(df_cv.to_string(index=False))
print(f"Mean Val Loss Ampl:  {df_cv['best_val_loss_ampl'].mean():.6e} ± {df_cv['best_val_loss_ampl'].std():.6e}")
print(f"Mean Val Loss Phase: {df_cv['best_val_loss_phase'].mean():.6e} ± {df_cv['best_val_loss_phase'].std():.6e}")

# Find fold with lowest combined validation loss
best_fold_idx = int(np.argmin(df_cv["best_val_loss_ampl"] + df_cv["best_val_loss_phase"])) + 1
print(f"\nEvaluating Best Fold (Fold {best_fold_idx}) on the held-out test set ({len(test_idx)} waveforms)...")

best_fold_ampl_path = f"{title}_fold{best_fold_idx}_amplitude_best.pt"
best_fold_phase_path = f"{title}_fold{best_fold_idx}_phase_best.pt"

eval_model_ampl, eval_model_phase = build_full_models()
eval_model_ampl.load_state_dict(torch.load(best_fold_ampl_path, map_location=device))
eval_model_phase.load_state_dict(torch.load(best_fold_phase_path, map_location=device))
eval_model_ampl.eval()
eval_model_phase.eval()

mismatches = []
with torch.no_grad():
    for b_params, b_ampl, b_phase in test_loader:
        b_params = b_params.float().to(device)
        b_ampl = b_ampl.float().to(device)
        b_phase = b_phase.float().to(device)

        pred_ampl = eval_model_ampl(b_params)
        pred_phase = eval_model_phase(b_params)

        true_wf = torch.polar(b_ampl, -b_phase * 100.0)
        pred_wf = torch.polar(torch.clamp(pred_ampl, min=0.0), -pred_phase * 100.0)

        match = TasosOverlapTorch(true_wf, pred_wf)
        mismatch = torch.clamp(1.0 - match, min=0.0)
        mismatches.extend(mismatch.detach().cpu().numpy().tolist())

mismatches = np.array(mismatches, dtype=np.float64)

plot_mismatch_filename = f"{title}_test_mismatch_distribution.png"
plot_mismatch_distribution(
    mismatches=mismatches,
    run_title=f"{title} (Best Fold {best_fold_idx})",
    output_path=plot_mismatch_filename,
)

print(f"\n=== Final Held-Out Test Set Mismatch Stats (Fold {best_fold_idx}) ===")
print(f"  Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
print(f"Artifacts saved:")
print(f"  - CV Summary CSV:  {title}_cv_summary.csv")
print(f"  - Test Mismatch:   {plot_mismatch_filename}")
