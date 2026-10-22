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



from models import Autoencoder3, Autoencoder_CNN_NOBN
from utils import (
    get_phase,
    train_one_epoch,
    validate,
    plot_training_curves,
    compute_mismatches,
    plot_mismatch_distribution,
    EarlyStopping,
)

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train 2 Autoencoders for Amplitude and Phase (NRHybSur3dq8)")
parser.add_argument("--title", type=str, default=None, help="Title / prefix for this run (used in output filenames)")
parser.add_argument("--model", type=str, choices=["autoencoder3", "autoencoder_cnn_nobn", "nobn"], default="nobn", help="Architecture: 'autoencoder3' (GroupNorm) or 'autoencoder_cnn_nobn'/'nobn' (No BatchNorm) (default: nobn)")
parser.add_argument("--opt", type=str, choices=["adam", "soap"], default="adam", help="Optimizer choice: 'adam' or 'soap' (default: adam)")
parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine"], default="plateau", help="LR scheduler choice: 'plateau' or 'cosine' (default: plateau)")
parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs (default: 3000)")
parser.add_argument("--early_stop", type=int, default=100, help="Early stopping patience in epochs (default: 100, set 0 to disable)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"AE_{args.opt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_name = "Autoencoder3 (GroupNorm)" if args.model.lower() == "autoencoder3" else "Autoencoder_CNN_NOBN"
print(f"=== Starting Run: {title} | Model: {model_name} | Optimizer: {args.opt.upper()} | Scheduler: {args.scheduler.upper()} | Epochs: {args.epochs} ===")

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
filepath = (
    "datasets/NRHybSur3dq8_dataset.hdf"
    if os.path.exists("datasets/NRHybSur3dq8_dataset.hdf")
    else "/lustre/ific.uv.es/ml/uv098/osvaldogramaxo/DANSur_22/data/NRHybSur3dq8_dataset.hdf"
)

with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    n_samples, sample_len = waveforms_ds.shape
    amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
    phase = np.empty((n_samples, sample_len), dtype=np.float32)
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        amplitude[start:end] = np.abs(waveforms_ds[start:end]).astype(np.float32)
        phase[start:end] = get_phase(waveforms_ds[start:end]).astype(np.float32)

ampl_tensor = torch.from_numpy(amplitude)
phase_tensor = torch.from_numpy(phase)

# Split train/val/test once
joint_dataset = TensorDataset(ampl_tensor, phase_tensor)
j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

# Derive single-branch subsets using the exact same indices
a_train = Subset(ampl_tensor, j_train.indices)
p_train = Subset(phase_tensor, j_train.indices)
a_val   = Subset(ampl_tensor, j_val.indices)
p_val   = Subset(phase_tensor, j_val.indices)

test_loader = DataLoader(j_test, batch_size=1024, shuffle=False)
a_train_loader = DataLoader(a_train, batch_size=64, shuffle=True)
a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False)
p_train_loader = DataLoader(p_train, batch_size=64, shuffle=True)
p_val_loader = DataLoader(p_val, batch_size=2048, shuffle=False)


# ---------------------------------------------------------------------------
# Models, Loss Functions & Regularization
# ---------------------------------------------------------------------------
if args.model.lower() == "autoencoder3":
    model_amplitude = Autoencoder3().to(device)
    model_phase = Autoencoder3().to(device)
else:
    model_amplitude = Autoencoder_CNN_NOBN().to(device)
    model_phase = Autoencoder_CNN_NOBN().to(device)

# Dedicated per-branch losses
loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3) #L1WeightLoss(w=5.0, start_window=1700, end_window=2048).to(device)
loss_fn_phase = nn.SmoothL1Loss(beta=1e-3) #L1WeightLoss(w=5.0, start_window=1700, end_window=2048).to(device)

lambda_latent = 1e-5

# ---------------------------------------------------------------------------
# Optimizers & Schedulers
# ---------------------------------------------------------------------------
if args.opt.lower() == "soap":
    opt_fn_amplitude = opt_add.SOAP(
        model_amplitude.parameters(), lr=4e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
    opt_fn_phase = opt_add.SOAP(
        model_phase.parameters(), lr=4e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
else:
    opt_fn_amplitude = AdamW(model_amplitude.parameters(), lr=4e-3, weight_decay=0)
    opt_fn_phase = AdamW(model_phase.parameters(), lr=4e-3, weight_decay=0)

if args.scheduler == "cosine":
    scheduler_amplitude = lr_scheduler.CosineAnnealingLR(
        opt_fn_amplitude, T_max=args.epochs, eta_min=1e-6
    )
    scheduler_phase = lr_scheduler.CosineAnnealingLR(
        opt_fn_phase, T_max=args.epochs, eta_min=1e-6
    )
elif args.opt.lower() == "soap":
    scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
        opt_fn_amplitude, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=15
    )
    scheduler_phase = lr_scheduler.ReduceLROnPlateau(
        opt_fn_phase, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=15
    )
else:
    scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
        opt_fn_amplitude, factor=0.5, patience=200, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=50
    )
    scheduler_phase = lr_scheduler.ReduceLROnPlateau(
        opt_fn_phase, factor=0.5, patience=200, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=50
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

# Early Stopping Initialization
if args.early_stop > 0:
    early_stop_amplitude = EarlyStopping(patience=args.early_stop, min_delta=1e-3, mode="rel", min_lr=1e-6)
    early_stop_phase = EarlyStopping(patience=args.early_stop, min_delta=1e-3, mode="rel", min_lr=1e-6)
    print(f"Early stopping enabled: patience={args.early_stop} epochs (relative delta=0.1%)")
else:
    print("Early stopping disabled.")

print(f"Starting training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    avg_loss_ampl = train_one_epoch(
        model_amplitude, opt_fn_amplitude, loss_fn_amplitude, device, epoch, None, a_train_loader, lambda_latent=lambda_latent
    )
    avg_loss_ph = train_one_epoch(
        model_phase, opt_fn_phase, loss_fn_phase, device, epoch, None, p_train_loader, lambda_latent=lambda_latent
    )

    # Validate
    avg_vloss_ampl = validate(model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=lambda_latent)
    avg_vloss_ph = validate(model_phase, p_val_loader, loss_fn_phase, device=device, lambda_latent=lambda_latent)

    # Step LR Scheduler
    if args.scheduler == "plateau":
        scheduler_amplitude.step(avg_vloss_ampl)
        scheduler_phase.step(avg_vloss_ph)
    else:
        scheduler_amplitude.step()
        scheduler_phase.step()

    current_lr_ampl = scheduler_amplitude.get_last_lr()[0]
    current_lr_ph = scheduler_phase.get_last_lr()[0]

    # Record history
    history["epoch"].append(epoch + 1)
    history["train_loss_amplitude"].append(avg_loss_ampl)
    history["val_loss_amplitude"].append(avg_vloss_ampl)
    history["lr_amplitude"].append(current_lr_ampl)
    history["train_loss_phase"].append(avg_loss_ph)
    history["val_loss_phase"].append(avg_vloss_ph)
    history["lr_phase"].append(current_lr_ph)

    # Save best Amplitude model & optimizer state
    if avg_vloss_ampl < best_vloss_amplitude:
        best_vloss_amplitude = avg_vloss_ampl
        torch.save(model_amplitude.state_dict(), f"{title}_amplitude_best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model_amplitude.state_dict(),
                "optimizer_state_dict": opt_fn_amplitude.state_dict(),
                "scheduler_state_dict": scheduler_amplitude.state_dict(),
                "best_vloss": best_vloss_amplitude,
            },
            f"{title}_amplitude_checkpoint.pt",
        )

    # Save best Phase model & optimizer state
    if avg_vloss_ph < best_vloss_phase:
        best_vloss_phase = avg_vloss_ph
        torch.save(model_phase.state_dict(), f"{title}_phase_best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model_phase.state_dict(),
                "optimizer_state_dict": opt_fn_phase.state_dict(),
                "scheduler_state_dict": scheduler_phase.state_dict(),
                "best_vloss": best_vloss_phase,
            },
            f"{title}_phase_checkpoint.pt",
        )

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"  Amplitude | Train Loss: {avg_loss_ampl:.6f} | Val Loss: {avg_vloss_ampl:.6f} | Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}")
        print(f"  Phase     | Train Loss: {avg_loss_ph:.6f} | Val Loss: {avg_vloss_ph:.6f} | Best: {best_vloss_phase:.6f} | LR: {current_lr_ph:.2e}")

    # Check Early Stopping
    if args.early_stop > 0:
        stop_a = early_stop_amplitude.step(avg_vloss_ampl, current_lr_ampl)
        stop_p = early_stop_phase.step(avg_vloss_ph, current_lr_ph)
        if stop_a and stop_p:
            print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
            print(f"  Amplitude: Best Val Loss {best_vloss_amplitude:.3e} (plateaued for {early_stop_amplitude.counter} epochs)")
            print(f"  Phase:     Best Val Loss {best_vloss_phase:.3e} (plateaued for {early_stop_phase.counter} epochs)")
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
model_amplitude.load_state_dict(torch.load(best_ampl_path, map_location=device))
model_phase.load_state_dict(torch.load(best_phase_path, map_location=device))

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
print(f"  - Amplitude Full Checkpoint: {title}_amplitude_checkpoint.pt (includes optimizer state)")
print(f"  - Best Phase Model:          {title}_phase_best.pt")
print(f"  - Phase Full Checkpoint:     {title}_phase_checkpoint.pt (includes optimizer state)")
print(f"  - History CSV:               {csv_filename}")
print(f"  - Amplitude Plot:            {plot_ampl_filename}")
print(f"  - Phase Plot:                {plot_phase_filename}")
print(f"  - Mismatch Plot:             {plot_mismatch_filename}")
print(f"  - Test Set Mismatch Stats:   Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
