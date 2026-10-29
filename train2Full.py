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
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
import pytorch_optimizer as opt_add

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

from models import (
    Autoencoder3,
    Autoencoder_CNN_NOBN,
    SimplestMapper,
    BiggerMapper,
    BiggestMapper,
    EvenBiggerMapper,
    FullModel,
)
from utils import (
    get_phase,
    train_one_epoch,
    validate,
    plot_training_curves,
    plot_mismatch_distribution,
    EarlyStopping,
    TasosOverlapTorch,
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
    """Detect mapper class from state_dict keys and shapes."""
    if "mapper.fc4.weight" in state_dict:
        fc1_out = state_dict["mapper.fc1.weight"].shape[0]
        if fc1_out == 64:
            return "even_bigger", EvenBiggerMapper
        elif fc1_out == 32:
            return "biggest", BiggestMapper
        elif fc1_out == 16:
            return "bigger", BiggerMapper
    elif "mapper.fc3.weight" in state_dict:
        return "simplest", SimplestMapper
    return None, None


def resolve_checkpoint_path(target, kind):
    """Find checkpoint file path for amplitude or phase."""
    if os.path.isfile(target):
        return target

    clean_target = target
    for suffix in [f"_{kind}_checkpoint.pt", f"_{kind}_best.pt", f"_{kind}.pt", ".pt"]:
        if clean_target.endswith(suffix):
            clean_target = clean_target[:-len(suffix)]
            break

    candidates = [
        f"{clean_target}_{kind}_checkpoint.pt",
        os.path.join("model_weights", f"{clean_target}_{kind}_checkpoint.pt"),
        os.path.join("model_weights", f"{os.path.basename(clean_target)}_{kind}_checkpoint.pt"),
        f"{clean_target}_{kind}_best.pt",
        os.path.join("model_weights", f"{clean_target}_{kind}_best.pt"),
        os.path.join("model_weights", f"{os.path.basename(clean_target)}_{kind}_best.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Could not locate checkpoint for '{kind}' with identifier '{target}'. Candidates checked: {candidates}"
    )


def resolve_history_csv(target):
    """Find matching metrics CSV file if it exists."""
    clean_target = target
    for suffix in ["_metrics.csv", ".csv"]:
        if clean_target.endswith(suffix):
            clean_target = clean_target[:-len(suffix)]
            break

    candidates = [
        f"{clean_target}_metrics.csv",
        os.path.join("runs", f"{clean_target}_metrics.csv"),
        os.path.join("runs", f"{os.path.basename(clean_target)}_metrics.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2 FullModels (Mapper + Frozen NOBN Decoder) for Amplitude and Phase (NRHybSur3dq8)")
    parser.add_argument("--title", type=str, default=None, help="Title / prefix for this run (used in output filenames)")
    parser.add_argument(
        "--mapper",
        type=str,
        default="simplest",
        choices=list(MAPPER_REGISTRY.keys()),
        help="Mapper architecture: 'simplest'/'simple', 'bigger'/'big', 'biggest', or 'even_bigger'/'huge' (default: simplest)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="AUTO",
        default=None,
        help="Resume training from checkpoint. Can be title/prefix (e.g. 'soapFull2628' or 'model_weights/soapFull2628') or checkpoint path. If passed without value, resumes from --title.",
    )
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine"], default="plateau", help="LR scheduler choice: 'plateau' or 'cosine' (default: plateau)")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs (default: 3000). If resuming and epochs <= checkpoint epoch, extends by this amount.")
    parser.add_argument("--early_stop", type=int, default=0, help="Early stopping patience in epochs (default: 0 disabled, e.g. 400)")
    return parser.parse_args()


def main():
    args = parse_args()

    opt_name = "SOAP"
    sched_name = args.scheduler.upper()

    resume_target = None
    if args.resume:
        resume_target = args.title if args.resume == "AUTO" else args.resume
        if not resume_target:
            raise ValueError("--resume was specified without a value, but --title was not provided either.")

    title = args.title
    if not title:
        if resume_target:
            title = f"{os.path.basename(resume_target)}_resumed"
        else:
            try:
                title = input("Enter title of this run: ").strip()
            except EOFError:
                title = ""
            if not title:
                title = f"{args.mapper.capitalize()}FM_{opt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mapper_key = args.mapper
    MapperClass = MAPPER_REGISTRY[mapper_key]

    print(f"=== Starting Run: {title} | Mapper: {MapperClass.__name__} | Optimizer: {opt_name} | Scheduler: {sched_name} | Epochs: {args.epochs} ===")
    if resume_target:
        print(f"=== Mode: RESUMING from '{resume_target}' ===")

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
        raw_params = f["Parameters"][:]
        waveforms_ds = f["Waveforms"]
        n_samples, sample_len = waveforms_ds.shape
        amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
        phase = np.empty((n_samples, sample_len), dtype=np.float32)
        chunk_size = 10000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            amplitude[start:end] = np.abs(waveforms_ds[start:end]).astype(np.float32)
            phase[start:end] = get_phase(waveforms_ds[start:end]).astype(np.float32)

    # Drop column 1 (constant total mass 1.0) -> [q, chi1, chi2]
    params = np.delete(raw_params, 1, axis=1).astype(np.float32)
    # Convert q to 1/q
    params[:, 0] = 1.0 / params[:, 0]

    params_tensor = torch.from_numpy(params)
    ampl_tensor = torch.from_numpy(amplitude)
    phase_tensor = torch.from_numpy(phase)

    # Split train/val/test once (reproducible seed 42)
    joint_dataset = TensorDataset(params_tensor, ampl_tensor, phase_tensor)
    j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    # Subsets pairing inputs (params) with targets (amplitude / phase)
    a_dataset = TensorDataset(params_tensor, ampl_tensor)
    p_dataset = TensorDataset(params_tensor, phase_tensor)

    a_train = Subset(a_dataset, j_train.indices)
    p_train = Subset(p_dataset, j_train.indices)
    a_val   = Subset(a_dataset, j_val.indices)
    p_val   = Subset(p_dataset, j_val.indices)

    test_loader = DataLoader(j_test, batch_size=1024, shuffle=False)
    a_train_loader = DataLoader(a_train, batch_size=128, shuffle=True)
    a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False)
    p_train_loader = DataLoader(p_train, batch_size=128, shuffle=True)
    p_val_loader = DataLoader(p_val, batch_size=2048, shuffle=False)

    # ---------------------------------------------------------------------------
    # Model Setup
    # ---------------------------------------------------------------------------
    # Pre-check checkpoint mapper architecture if resuming
    ckpt_ampl = None
    ckpt_phase = None
    if resume_target:
        ampl_ckpt_path = resolve_checkpoint_path(resume_target, "amplitude")
        phase_ckpt_path = resolve_checkpoint_path(resume_target, "phase")
        ckpt_ampl = torch.load(ampl_ckpt_path, map_location=device)
        ckpt_phase = torch.load(phase_ckpt_path, map_location=device)

        det_name, det_cls = detect_mapper_from_state(ckpt_ampl.get("model_state_dict", ckpt_ampl))
        if det_cls and det_cls != MapperClass:
            print(f"Checkpoint uses mapper '{det_name}' ({det_cls.__name__}), auto-switching mapper from '{args.mapper}'.")
            MapperClass = det_cls

    model_amplitude = MapperClass().to(device)
    model_phase = MapperClass().to(device)

    ae_amplitude = Autoencoder_CNN_NOBN().to(device)
    ae_phase = Autoencoder_CNN_NOBN().to(device)

    ae_ampl_path = "model_weights/soapNOBN2610_amplitude_best.pt" if os.path.exists("model_weights/soapNOBN2610_amplitude_best.pt") else "soapNOBN2610_amplitude_best.pt"
    ae_phase_path = "model_weights/soapNOBN2610_phase_best.pt" if os.path.exists("model_weights/soapNOBN2610_phase_best.pt") else "soapNOBN2610_phase_best.pt"

    ae_amplitude.load_state_dict(torch.load(ae_ampl_path, map_location=device))
    ae_phase.load_state_dict(torch.load(ae_phase_path, map_location=device))

    filter_amplitude = ae_amplitude.decoder.eval()
    filter_phase = ae_phase.decoder.eval()

    filter_amplitude.requires_grad_(False)
    filter_phase.requires_grad_(False)

    full_model_ampl = FullModel(model_amplitude, filter_amplitude).to(device)
    full_model_phase = FullModel(model_phase, filter_phase).to(device)

    loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3)
    loss_fn_phase = nn.SmoothL1Loss(beta=1e-3)

    opt_fn_amplitude = opt_add.SOAP(
        model_amplitude.parameters(), lr=2e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
    opt_fn_phase = opt_add.SOAP(
        model_phase.parameters(), lr=2e-3, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )

    if args.scheduler == "cosine":
        scheduler_amplitude = lr_scheduler.CosineAnnealingLR(
            opt_fn_amplitude, T_max=args.epochs, eta_min=5e-7
        )
        scheduler_phase = lr_scheduler.CosineAnnealingLR(
            opt_fn_phase, T_max=args.epochs, eta_min=5e-7
        )
    else:
        scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
            opt_fn_amplitude, factor=0.5, patience=100, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=50
        )
        scheduler_phase = lr_scheduler.ReduceLROnPlateau(
            opt_fn_phase, factor=0.5, patience=100, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=50
        )

    best_vloss_amplitude = float("inf")
    best_vloss_phase = float("inf")
    start_epoch = 0
    total_epochs = args.epochs

    history = {
        "epoch": [],
        "train_loss_amplitude": [],
        "val_loss_amplitude": [],
        "lr_amplitude": [],
        "train_loss_phase": [],
        "val_loss_phase": [],
        "lr_phase": [],
    }

    # ---------------------------------------------------------------------------
    # Resume Checkpoints
    # ---------------------------------------------------------------------------
    if resume_target:
        print(f"Loading checkpoint states:")
        print(f"  Amplitude: {ampl_ckpt_path}")
        print(f"  Phase:     {phase_ckpt_path}")

        full_model_ampl.load_state_dict(ckpt_ampl["model_state_dict"])
        full_model_phase.load_state_dict(ckpt_phase["model_state_dict"])

        if "optimizer_state_dict" in ckpt_ampl:
            opt_fn_amplitude.load_state_dict(ckpt_ampl["optimizer_state_dict"])
        if "optimizer_state_dict" in ckpt_phase:
            opt_fn_phase.load_state_dict(ckpt_phase["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt_ampl:
            scheduler_amplitude.load_state_dict(ckpt_ampl["scheduler_state_dict"])
        if "scheduler_state_dict" in ckpt_phase:
            scheduler_phase.load_state_dict(ckpt_phase["scheduler_state_dict"])

        best_vloss_amplitude = ckpt_ampl.get("best_vloss", float("inf"))
        best_vloss_phase = ckpt_phase.get("best_vloss", float("inf"))

        epoch_ampl = ckpt_ampl.get("epoch", 0)
        epoch_phase = ckpt_phase.get("epoch", 0)
        start_epoch = min(epoch_ampl, epoch_phase)

        if total_epochs <= start_epoch:
            total_epochs = start_epoch + args.epochs
            print(f"Target epochs ({args.epochs}) <= checkpoint epoch ({start_epoch}). Extending total epochs to {total_epochs}.")
        else:
            print(f"Resuming from epoch {start_epoch} up to target {total_epochs} epochs.")

        print(f"Resumed baseline validation losses: Amplitude={best_vloss_amplitude:.3e}, Phase={best_vloss_phase:.3e}")

        # Load previous history CSV if found
        csv_prev = resolve_history_csv(resume_target)
        if csv_prev:
            try:
                df_prev = pd.read_csv(csv_prev)
                df_prev = df_prev[df_prev["epoch"] <= start_epoch]
                for col in history.keys():
                    if col in df_prev.columns:
                        history[col] = df_prev[col].tolist()
                print(f"Loaded {len(history['epoch'])} rows of previous metrics history from {csv_prev}.")
            except Exception as e:
                print(f"Could not load previous metrics history from {csv_prev}: {e}")

    # ---------------------------------------------------------------------------
    # Early Stopping Initialization
    # ---------------------------------------------------------------------------
    if args.early_stop > 0:
        early_stop_amplitude = EarlyStopping(
            patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-6, patience_at_min_lr=100
        )
        early_stop_phase = EarlyStopping(
            patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-6, patience_at_min_lr=100
        )
        if resume_target:
            early_stop_amplitude.best_loss = best_vloss_amplitude
            early_stop_phase.best_loss = best_vloss_phase
        print(f"Early stopping enabled: global_patience={args.early_stop}, floor_patience=100 at min_lr=1e-6 (rel threshold=0.01%)")
    else:
        print("Early stopping disabled.")

    # ---------------------------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------------------------
    print(f"\n--- Training from epoch {start_epoch + 1} to {total_epochs} ---")

    for epoch in range(start_epoch, total_epochs):
        model_amplitude.train(True)
        total_loss_amplitudeT = 0.0

        for i, (p_in, ampl) in enumerate(a_train_loader):
            p_in, ampl = p_in.to(device), ampl.to(device)
            opt_fn_amplitude.zero_grad()
            pred_latent_ampl = model_amplitude(p_in)
            pred_wf_ampl = filter_amplitude(pred_latent_ampl)
            loss_amp = loss_fn_amplitude(pred_wf_ampl.squeeze(1), ampl)
            loss_amp.backward()
            opt_fn_amplitude.step()
            total_loss_amplitudeT += loss_amp.item()
        total_loss_amplitudeT /= max(len(a_train_loader), 1)

        model_amplitude.eval()
        total_loss_amplitudeV = 0.0
        for i, (p_in, ampl) in enumerate(a_val_loader):
            p_in, ampl = p_in.to(device), ampl.to(device)
            with torch.no_grad():
                pred_latent_ampl = model_amplitude(p_in)
                pred_wf_ampl = filter_amplitude(pred_latent_ampl)
                loss_amp = loss_fn_amplitude(pred_wf_ampl.squeeze(1), ampl)
                total_loss_amplitudeV += loss_amp.item()
        total_loss_amplitudeV /= max(len(a_val_loader), 1)

        if args.scheduler == "plateau":
            scheduler_amplitude.step(total_loss_amplitudeV)
        else:
            scheduler_amplitude.step()

        model_phase.train(True)
        total_loss_phaseT = 0.0
        for i, (p_in, ph) in enumerate(p_train_loader):
            p_in, ph = p_in.to(device), ph.to(device)
            opt_fn_phase.zero_grad()
            pred_latent_phase = model_phase(p_in)
            pred_wf_phase = filter_phase(pred_latent_phase)
            loss_phase = loss_fn_phase(pred_wf_phase.squeeze(1), ph)
            loss_phase.backward()
            opt_fn_phase.step()
            total_loss_phaseT += loss_phase.item()
        total_loss_phaseT /= max(len(p_train_loader), 1)

        model_phase.eval()
        total_loss_phaseV = 0.0
        for i, (p_in, ph) in enumerate(p_val_loader):
            p_in, ph = p_in.to(device), ph.to(device)
            with torch.no_grad():
                pred_latent_phase = model_phase(p_in)
                pred_wf_phase = filter_phase(pred_latent_phase)
                loss_phase = loss_fn_phase(pred_wf_phase.squeeze(1), ph)
                total_loss_phaseV += loss_phase.item()
        total_loss_phaseV /= max(len(p_val_loader), 1)

        if args.scheduler == "plateau":
            scheduler_phase.step(total_loss_phaseV)
        else:
            scheduler_phase.step()

        current_lr_ampl = scheduler_amplitude.get_last_lr()[0]
        current_lr_ph = scheduler_phase.get_last_lr()[0]

        history["epoch"].append(epoch + 1)
        history["train_loss_amplitude"].append(total_loss_amplitudeT)
        history["val_loss_amplitude"].append(total_loss_amplitudeV)
        history["lr_amplitude"].append(current_lr_ampl)
        history["train_loss_phase"].append(total_loss_phaseT)
        history["val_loss_phase"].append(total_loss_phaseV)
        history["lr_phase"].append(current_lr_ph)

        if total_loss_amplitudeV < best_vloss_amplitude:
            best_vloss_amplitude = total_loss_amplitudeV
            torch.save(full_model_ampl.state_dict(), f"{title}_amplitude_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": full_model_ampl.state_dict(),
                    "optimizer_state_dict": opt_fn_amplitude.state_dict(),
                    "scheduler_state_dict": scheduler_amplitude.state_dict(),
                    "best_vloss": best_vloss_amplitude,
                },
                f"{title}_amplitude_checkpoint.pt",
            )

        if total_loss_phaseV < best_vloss_phase:
            best_vloss_phase = total_loss_phaseV
            torch.save(full_model_phase.state_dict(), f"{title}_phase_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": full_model_phase.state_dict(),
                    "optimizer_state_dict": opt_fn_phase.state_dict(),
                    "scheduler_state_dict": scheduler_phase.state_dict(),
                    "best_vloss": best_vloss_phase,
                },
                f"{title}_phase_checkpoint.pt",
            )

        if (epoch + 1) % 50 == 0 or epoch == start_epoch or (epoch + 1) == total_epochs:
            print(
                f"Epoch [{epoch + 1}/{total_epochs}] "
                f"Ampl: train={total_loss_amplitudeT:.3e} val={total_loss_amplitudeV:.3e} (best={best_vloss_amplitude:.3e}, lr={current_lr_ampl:.1e}) | "
                f"Phase: train={total_loss_phaseT:.3e} val={total_loss_phaseV:.3e} (best={best_vloss_phase:.3e}, lr={current_lr_ph:.1e})"
            )

        # Check Early Stopping
        if args.early_stop > 0:
            stop_a = early_stop_amplitude.step(total_loss_amplitudeV, current_lr_ampl)
            stop_p = early_stop_phase.step(total_loss_phaseV, current_lr_ph)
            if stop_a and stop_p:
                print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
                print(f"  Amplitude: {early_stop_amplitude.stop_reason}")
                print(f"  Phase:     {early_stop_phase.stop_reason}")
                break

    # ---------------------------------------------------------------------------
    # Save 1 Unified CSV History
    # ---------------------------------------------------------------------------
    csv_filename = f"{title}_metrics.csv"
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_filename, index=False)
    print(f"\nUnified metrics CSV saved: {csv_filename}")

    # ---------------------------------------------------------------------------
    # Generate Training Curves
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
    full_model_ampl.load_state_dict(torch.load(best_ampl_path, map_location=device))
    full_model_phase.load_state_dict(torch.load(best_phase_path, map_location=device))
    full_model_ampl.eval()
    full_model_phase.eval()

    print("Computing test set waveform mismatches...")
    mismatches = []
    with torch.no_grad():
        for b_params, b_ampl, b_phase in test_loader:
            b_params = b_params.float().to(device)
            b_ampl = b_ampl.float().to(device)
            b_phase = b_phase.float().to(device)

            pred_ampl = full_model_ampl(b_params)
            pred_phase = full_model_phase(b_params)

            true_wf = torch.polar(b_ampl, -b_phase * 100.0)
            pred_wf = torch.polar(torch.clamp(pred_ampl, min=0.0), -pred_phase * 100.0)

            match = TasosOverlapTorch(true_wf, pred_wf)
            mismatch = torch.clamp(1.0 - match, min=0.0)
            mismatches.extend(mismatch.detach().cpu().numpy().tolist())

    mismatches = np.array(mismatches, dtype=np.float64)

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


if __name__ == "__main__":
    main()
