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

from models import (
    SimplestMapper,
    BiggerMapper,
    BiggestMapper,
    EvenBiggerMapper,
    DecoderNOBN,
    FullModel,
)

MAPPER_REGISTRY = {
    "simplest": SimplestMapper,
    "simple": SimplestMapper,
    "bigger": BiggerMapper,
    "big": BiggerMapper,
    "biggest": BiggestMapper,
    "even_bigger": EvenBiggerMapper,
    "evenbigger": EvenBiggerMapper,
    "huge": EvenBiggerMapper,
}


def detect_mapper_from_state(state_dict):
    if "mapper.fc4.weight" in state_dict or "fc4.weight" in state_dict:
        fc1_w = state_dict.get("mapper.fc1.weight", state_dict.get("fc1.weight"))
        if fc1_w is not None:
            if fc1_w.shape[0] == 64:
                return "even_bigger", EvenBiggerMapper
            elif fc1_w.shape[0] == 32:
                return "biggest", BiggestMapper
            elif fc1_w.shape[0] == 16:
                return "bigger", BiggerMapper
    elif "mapper.fc3.weight" in state_dict or "fc3.weight" in state_dict:
        return "simplest", SimplestMapper
    return "simplest", SimplestMapper
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
# Hyperparameters & Defaults
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
K_FOLDS = 5

# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Single-Phase Joint Finetuning of FullModels with K-Fold CV on SXS")
parser.add_argument("--title", type=str, default=None, help="Title / prefix for this run")
parser.add_argument(
    "--mapper",
    type=str,
    default="auto",
    choices=["auto"] + list(MAPPER_REGISTRY.keys()),
    help="Mapper architecture (default: auto, detected from checkpoint)",
)
parser.add_argument(
    "--checkpoint_ampl",
    type=str,
    default="model_weights/soapFull2628_amplitude_best.pt",
    help="Path to pretrained amplitude checkpoint (default: model_weights/soapFull2628_amplitude_best.pt)",
)
parser.add_argument(
    "--checkpoint_phase",
    type=str,
    default="model_weights/soapFull2628_phase_best.pt",
    help="Path to pretrained phase checkpoint (default: model_weights/soapFull2628_phase_best.pt)",
)
parser.add_argument("--epochs", type=int, default=1500, help="Total training epochs per fold (default: 1500)")
parser.add_argument("--lr_mapper", type=float, default=1e-4, help="Learning rate for mapper (default: 1e-4)")
parser.add_argument("--lr_decoder", type=float, default=1e-5, help="Learning rate for decoder (default: 1e-5)")
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

print(f"=== Starting 5-Fold Joint Finetuning: {title} | Epochs: {args.epochs} | Mapper LR: {args.lr_mapper} | Decoder LR: {args.lr_decoder} ===")


def load_checkpoint_weights(model: FullModel, ckpt_path: str):
    """Loads weights into FullModel supporting FullModel, mapper-only, or AE state dicts."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found.")
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    if any(k.startswith("mapper.") for k in sd.keys()):
        model.load_state_dict(sd)
        print(f"Loaded FullModel weights from: {ckpt_path}")
    elif any(k.startswith("fc1.") for k in sd.keys()):
        model.mapper.load_state_dict(sd)
        print(f"Loaded mapper weights from: {ckpt_path}")
    elif any(k.startswith("decoder.") for k in sd.keys()) or any(k.startswith("encoder.") for k in sd.keys()):
        model.decoder.load_state_dict(sd, strict=False)
        print(f"[Warning] Loaded decoder only from AE checkpoint: {ckpt_path}")
    else:
        model.load_state_dict(sd, strict=False)
        print(f"Loaded weights with strict=False from: {ckpt_path}")


def build_full_models(checkpoint_ampl: str, checkpoint_phase: str, mapper_name: str = "auto"):
    """Initializes FullModels and reloads pretrained mapper and decoder weights."""
    if not os.path.exists(checkpoint_ampl):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_ampl}' not found.")
    if not os.path.exists(checkpoint_phase):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_phase}' not found.")

    if mapper_name == "auto":
        sd_a = torch.load(checkpoint_ampl, map_location=device)
        if isinstance(sd_a, dict) and "model_state_dict" in sd_a:
            sd_a = sd_a["model_state_dict"]
        det_name, MapperClass = detect_mapper_from_state(sd_a)
        print(f"Auto-detected mapper '{det_name}' ({MapperClass.__name__}) from checkpoint.")
    else:
        MapperClass = MAPPER_REGISTRY[mapper_name]

    model_amplitude = MapperClass().to(device)
    model_phase = MapperClass().to(device)
    decoder_amplitude = DecoderNOBN().to(device)
    decoder_phase = DecoderNOBN().to(device)

    full_model_ampl = FullModel(model_amplitude, decoder_amplitude).to(device)
    full_model_phase = FullModel(model_phase, decoder_phase).to(device)

    load_checkpoint_weights(full_model_ampl, checkpoint_ampl)
    load_checkpoint_weights(full_model_phase, checkpoint_phase)
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
    full_model_ampl, full_model_phase = build_full_models(args.checkpoint_ampl, args.checkpoint_phase, args.mapper)

    loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3)
    loss_fn_phase = nn.SmoothL1Loss(beta=1e-3)

    # Single-Phase Joint Optimization (Mapper + Decoder unfrozen from Epoch 1)
    full_model_ampl.mapper.requires_grad_(True)
    full_model_ampl.decoder.requires_grad_(True)
    full_model_phase.mapper.requires_grad_(True)
    full_model_phase.decoder.requires_grad_(True)

    opt_fn_amplitude = opt_add.SOAP([
        {"params": full_model_ampl.mapper.parameters(), "lr": args.lr_mapper},
        {"params": full_model_ampl.decoder.parameters(), "lr": args.lr_decoder},
    ], betas=(0.95, 0.999), shampoo_beta=0.99, weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001)

    opt_fn_phase = opt_add.SOAP([
        {"params": full_model_phase.mapper.parameters(), "lr": args.lr_mapper},
        {"params": full_model_phase.decoder.parameters(), "lr": args.lr_decoder},
    ], betas=(0.95, 0.999), shampoo_beta=0.99, weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001)

    print(f"[Fold {fold_num}] Single-Phase Joint Training: Mapper LR={args.lr_mapper}, Decoder LR={args.lr_decoder}")

    scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
        opt_fn_amplitude, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-7, cooldown=25
    )
    scheduler_phase = lr_scheduler.ReduceLROnPlateau(
        opt_fn_phase, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-7, cooldown=25
    )

    if args.early_stop > 0:
        early_stop_amplitude = EarlyStopping(
            patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-7, patience_at_min_lr=50
        )
        early_stop_phase = EarlyStopping(
            patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-7, patience_at_min_lr=50
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

        # Periodic Progress Output
        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(
                f"[Fold {fold_num}] Epoch [{epoch + 1}/{args.epochs}] "
                f"Ampl: train={total_loss_amplitudeT:.3e} val={total_loss_amplitudeV:.3e} (best={best_vloss_amplitude:.3e}) | "
                f"Phase: train={total_loss_phaseT:.3e} val={total_loss_phaseV:.3e} (best={best_vloss_phase:.3e})"
            )

        # 5. Early Stopping Check
        if args.early_stop > 0 and epoch >= 50:
            stop_a = early_stop_amplitude.step(total_loss_amplitudeV, current_lr_ampl)
            stop_p = early_stop_phase.step(total_loss_phaseV, current_lr_ph)
            if stop_a and stop_p:
                print(f"[Fold {fold_num}] Early Stopping Triggered at Epoch {epoch + 1} (Ampl: {early_stop_amplitude.stop_reason}, Phase: {early_stop_phase.stop_reason})")
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

eval_model_ampl, eval_model_phase = build_full_models(args.checkpoint_ampl, args.checkpoint_phase, args.mapper)
eval_model_ampl.load_state_dict(torch.load(best_fold_ampl_path, map_location=device))
eval_model_phase.load_state_dict(torch.load(best_fold_phase_path, map_location=device))
eval_model_ampl.eval()
eval_model_phase.eval()

# Save single definitive winning model
best_ampl_out = f"{title}_amplitude_best.pt"
best_phase_out = f"{title}_phase_best.pt"
torch.save(eval_model_ampl.state_dict(), best_ampl_out)
torch.save(eval_model_phase.state_dict(), best_phase_out)
print(f"\nSaved single best model (from Fold {best_fold_idx}):")
print(f"  - {best_ampl_out}")
print(f"  - {best_phase_out}")

# Clean up intermediate fold .pt files so only 1 model is retained
for f_idx in range(1, k + 1):
    for suffix in ["_amplitude_best.pt", "_amplitude_checkpoint.pt", "_phase_best.pt", "_phase_checkpoint.pt"]:
        fold_ckpt = f"{title}_fold{f_idx}{suffix}"
        if os.path.exists(fold_ckpt):
            os.remove(fold_ckpt)

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
print(f"  - Best Amplitude Model: {best_ampl_out}")
print(f"  - Best Phase Model:     {best_phase_out}")
print(f"  - CV Summary CSV:       {title}_cv_summary.csv")
print(f"  - Test Mismatch:        {plot_mismatch_filename}")
