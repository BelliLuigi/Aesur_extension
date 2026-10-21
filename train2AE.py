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
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_optimizer as opt_add

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False



from models import Autoencoder_CNN_1, Autoencoder_CNN_NOBN
from utils import (
    get_phase,
    train_one_epoch,
    train_one_epoch_maesmatch,
    L1WeightLoss,
    AESurLoss,
    validate,
    validate_maesmatch,
    plot_training_curves,
    compute_mismatches,
    plot_mismatch_distribution,
)

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train 2 Autoencoders for Amplitude and Phase (NRHybSur3dq8)")
parser.add_argument("--title", type=str, default=None, help="Title / prefix for this run (used in output filenames)")
parser.add_argument("--opt", type=str, choices=["adam", "soap"], default="adam", help="Optimizer choice: 'adam' or 'soap' (default: adam)")
parser.add_argument("--loss", type=str, choices=["mae", "maesmatch"], default="mae", help="Loss choice: 'mae' (separate L1WeightLoss) or 'maesmatch' (hybrid direct L1WeightLoss + joint AESur composite loss)")
parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs (default: 3000)")
parser.add_argument("--lambda_joint", type=float, default=1e-3, help="Weight of the joint complex waveform maesmatch loss (default: 0.1)")
args = parser.parse_args()

title = args.title
if not title:
    try:
        title = input("Enter title of this run: ").strip()
    except EOFError:
        title = ""
    if not title:
        title = f"AE_{args.opt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"=== Starting Run: {title} | Optimizer: {args.opt.upper()} | Loss: {args.loss.upper()} | Epochs: {args.epochs} ===")

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

# Split train/val/test
joint_dataset = TensorDataset(ampl_tensor, phase_tensor)
j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(j_test, batch_size=1024, shuffle=False)

if args.loss == "maesmatch":
    train_loader = DataLoader(j_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(j_val, batch_size=2048, shuffle=False)
else:
    a_train, a_val, _ = random_split(ampl_tensor, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    p_train, p_val, _ = random_split(phase_tensor, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    a_train_loader = DataLoader(a_train, batch_size=64, shuffle=True)
    a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False)

    p_train_loader = DataLoader(p_train, batch_size=64, shuffle=True)
    p_val_loader = DataLoader(p_val, batch_size=2048, shuffle=False)

# ---------------------------------------------------------------------------
# Models, Loss Functions & Regularization
# ---------------------------------------------------------------------------
model_amplitude = Autoencoder_CNN_NOBN().to(device)#Autoencoder_CNN_1().to(device)
model_phase = Autoencoder_CNN_NOBN().to(device)#Autoencoder_CNN_1().to(device)

# Dedicated per-branch losses (used in both 'mae' and 'maesmatch' modes)
loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3) #L1WeightLoss(w=5.0, start_window=1700, end_window=2048).to(device)
loss_fn_phase = nn.SmoothL1Loss(beta=1e-3) #L1WeightLoss(w=5.0, start_window=1700, end_window=2048).to(device)

if args.loss == "maesmatch":
    loss_fn_joint = AESurLoss().to(device)

lambda_latent = 1e-5
lambda_joint = args.lambda_joint

# ---------------------------------------------------------------------------
# Optimizers & Schedulers
# ---------------------------------------------------------------------------
if args.opt.lower() == "soap":
    opt_fn_amplitude = opt_add.SOAP(
        model_amplitude.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=1e-9, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
    opt_fn_phase = opt_add.SOAP(
        model_phase.parameters(), lr=5e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=1e-9, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
else:
    opt_fn_amplitude = AdamW(model_amplitude.parameters(), lr=7e-3, weight_decay=1e-9)
    opt_fn_phase = AdamW(model_phase.parameters(), lr=7e-3, weight_decay=1e-9)

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
    "train_loss_joint": [],
    "val_loss_joint": [],
}

print(f"Starting training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    if args.loss == "maesmatch":
        avg_loss_ampl, avg_loss_ph, avg_loss_joint = train_one_epoch_maesmatch(
            model_amplitude,
            model_phase,
            opt_fn_amplitude,
            opt_fn_phase,
            loss_fn_joint,
            loss_fn_amplitude,
            loss_fn_phase,
            device,
            epoch,
            None,
            train_loader,
            lambda_latent=lambda_latent,
            lambda_joint=lambda_joint,
        )
        avg_vloss_ampl, avg_vloss_ph, avg_vloss_joint = validate_maesmatch(
            model_amplitude,
            model_phase,
            val_loader,
            loss_fn_joint,
            loss_fn_amplitude,
            loss_fn_phase,
            device,
            lambda_latent=lambda_latent,
            lambda_joint=lambda_joint,
        )

        # Step schedulers independently on each branch's composite validation loss
        scheduler_amplitude.step(avg_vloss_ampl)
        scheduler_phase.step(avg_vloss_ph)

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
        history["train_loss_joint"].append(avg_loss_joint)
        history["val_loss_joint"].append(avg_vloss_joint)

        # Save best Amplitude model & optimizer state independently
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

        # Save best Phase model & optimizer state independently
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
            print(f"Epoch [{epoch + 1}/{args.epochs}] (MAESMatch Hybrid | Joint WF Val: {avg_vloss_joint:.6f})")
            print(f"  Amplitude | Train: {avg_loss_ampl:.6f} | Val: {avg_vloss_ampl:.6f} | Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}")
            print(f"  Phase     | Train: {avg_loss_ph:.6f} | Val: {avg_vloss_ph:.6f} | Best: {best_vloss_phase:.6f} | LR: {current_lr_ph:.2e}")

    else:
        # Original MAE separate training
        avg_loss_ampl = train_one_epoch(
            model_amplitude, opt_fn_amplitude, loss_fn_amplitude, device, epoch, None, a_train_loader, lambda_latent=lambda_latent
        )
        avg_loss_ph = train_one_epoch(
            model_phase, opt_fn_phase, loss_fn_phase, device, epoch, None, p_train_loader, lambda_latent=lambda_latent
        )

        # Validate
        avg_vloss_ampl = validate(model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=lambda_latent)
        scheduler_amplitude.step(avg_vloss_ampl)

        avg_vloss_ph = validate(model_phase, p_val_loader, loss_fn_phase, device=device, lambda_latent=lambda_latent)
        scheduler_phase.step(avg_vloss_ph)

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
        history["train_loss_joint"].append(0.0)
        history["val_loss_joint"].append(0.0)

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
