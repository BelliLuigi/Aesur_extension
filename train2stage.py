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
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

from models import Autoencoder_CNN_NOBN, BiggestMapper, FullModel
from utils import (
    get_phase,
    plot_training_curves,
    plot_mismatch_distribution,
    EarlyStopping,
    TasosOverlapTorch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage End-to-End Training on NRHybSur3dq8: Frozen Decoder Warmup -> Joint Refinement"
    )
    parser.add_argument("--title", type=str, default=None, help="Run title / output prefix")
    parser.add_argument(
        "--ae_ampl",
        type=str,
        default="model_weights/soapNOBN2610_amplitude_best.pt",
        help="Path to pre-trained amplitude Autoencoder weights",
    )
    parser.add_argument(
        "--ae_phase",
        type=str,
        default="model_weights/soapNOBN2610_phase_best.pt",
        help="Path to pre-trained phase Autoencoder weights",
    )
    # Stage 1 (Frozen Decoder Warmup)
    parser.add_argument("--stage1_max_epochs", type=int, default=1500, help="Max epochs for Stage 1 (default: 1500)")
    parser.add_argument("--stage1_lr", type=float, default=2e-3, help="Stage 1 mapper learning rate (default: 2e-3)")
    parser.add_argument(
        "--stage1_trigger",
        type=str,
        choices=["auto", "lr", "epochs"],
        default="auto",
        help="Stage 2 transition trigger: 'auto' (LR drop or max epochs), 'lr' (threshold only), 'epochs' (fixed epochs)",
    )
    parser.add_argument(
        "--stage1_lr_threshold",
        type=float,
        default=2.5e-4,
        help="LR floor to trigger Stage 2 when stage1_trigger is 'auto' or 'lr' (default: 2.5e-4)",
    )
    # Stage 2 (Joint Refinement)
    parser.add_argument("--stage2_epochs", type=int, default=1000, help="Max epochs for Stage 2 (default: 1000)")
    parser.add_argument("--stage2_lr_mapper", type=float, default=1e-4, help="Stage 2 mapper LR (default: 1e-4)")
    parser.add_argument("--stage2_lr_decoder", type=float, default=1e-5, help="Stage 2 decoder LR (default: 1e-5)")
    parser.add_argument("--early_stop", type=int, default=200, help="Stage 2 early stopping patience (default: 100, 0 to disable)")
    return parser.parse_args()


def main():
    args = parse_args()

    title = args.title
    if not title:
        try:
            title = input("Enter title of this run: ").strip()
        except EOFError:
            title = ""
        if not title:
            title = f"TwoStage_Biggest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"=== Two-Stage NRHyb Training: {title} ===")
    print(f"  Architecture: BiggestMapper (Scratch) + DecoderNOBN (Pretrained)")
    print(f"  Stage 1: Frozen Decoder | LR={args.stage1_lr} | Max Epochs={args.stage1_max_epochs} | Trigger={args.stage1_trigger} (LR threshold={args.stage1_lr_threshold})")
    print(f"  Stage 2: Joint Unfrozen  | Mapper LR={args.stage2_lr_mapper} | Decoder LR={args.stage2_lr_decoder} | Max Epochs={args.stage2_epochs} | EarlyStop={args.early_stop}")

    # ---------------------------------------------------------------------------
    # Device Configuration
    # ---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        print(f"Allocated GPU: {prop.name} | VRAM: {prop.total_memory / (1024**3):.1f} GB")
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

    params = np.delete(raw_params, 1, axis=1).astype(np.float32)
    params[:, 0] = 1.0 / params[:, 0]

    params_tensor = torch.from_numpy(params)
    ampl_tensor = torch.from_numpy(amplitude)
    phase_tensor = torch.from_numpy(phase)

    joint_dataset = TensorDataset(params_tensor, ampl_tensor, phase_tensor)
    j_train, j_val, j_test = random_split(joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

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
    # Model Initialization (Scratch BiggestMapper + Pretrained AE Decoder)
    # ---------------------------------------------------------------------------
    model_amplitude = BiggestMapper().to(device)
    model_phase = BiggestMapper().to(device)

    ae_amplitude = Autoencoder_CNN_NOBN().to(device)
    ae_phase = Autoencoder_CNN_NOBN().to(device)

    ae_a_path = args.ae_ampl if os.path.exists(args.ae_ampl) else os.path.join("model_weights", os.path.basename(args.ae_ampl))
    ae_p_path = args.ae_phase if os.path.exists(args.ae_phase) else os.path.join("model_weights", os.path.basename(args.ae_phase))

    print(f"Loading pretrained decoders:")
    print(f"  Amplitude AE: {ae_a_path}")
    print(f"  Phase AE:     {ae_p_path}")
    ae_amplitude.load_state_dict(torch.load(ae_a_path, map_location=device))
    ae_phase.load_state_dict(torch.load(ae_p_path, map_location=device))

    filter_amplitude = ae_amplitude.decoder
    filter_phase = ae_phase.decoder

    full_model_ampl = FullModel(model_amplitude, filter_amplitude).to(device)
    full_model_phase = FullModel(model_phase, filter_phase).to(device)

    loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3)
    loss_fn_phase = nn.SmoothL1Loss(beta=1e-3)

    # ---------------------------------------------------------------------------
    # Stage 1 Setup: Frozen Decoder Warmup
    # ---------------------------------------------------------------------------
    current_stage = 1
    full_model_ampl.decoder.requires_grad_(False)
    full_model_phase.decoder.requires_grad_(False)
    full_model_ampl.mapper.requires_grad_(True)
    full_model_phase.mapper.requires_grad_(True)

    opt_amplitude = opt_add.SOAP(
        model_amplitude.parameters(), lr=args.stage1_lr, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )
    opt_phase = opt_add.SOAP(
        model_phase.parameters(), lr=args.stage1_lr, betas=(0.95, 0.999), shampoo_beta=0.99,
        weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
    )

    scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
        opt_amplitude, factor=0.5, patience=75, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=25
    )
    scheduler_phase = lr_scheduler.ReduceLROnPlateau(
        opt_phase, factor=0.5, patience=75, threshold=0.0001, threshold_mode="rel", min_lr=1e-6, cooldown=25
    )

    best_vloss_amplitude = float("inf")
    best_vloss_phase = float("inf")

    history = {
        "epoch": [],
        "stage": [],
        "train_loss_amplitude": [],
        "val_loss_amplitude": [],
        "lr_amplitude": [],
        "train_loss_phase": [],
        "val_loss_phase": [],
        "lr_phase": [],
    }

    early_stop_amplitude = EarlyStopping(
        patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-7, patience_at_min_lr=50
    )
    early_stop_phase = EarlyStopping(
        patience=args.early_stop, threshold=0.0001, mode="rel", min_lr=1e-7, patience_at_min_lr=50
    )

    stage1_completed_epochs = 0
    total_epoch_count = 0
    stage2_start_epoch = 0

    print(f"\n=======================================================")
    print(f"=== [STARTING STAGE 1: Frozen Decoder Warmup] ===")
    print(f"=======================================================")

    # ---------------------------------------------------------------------------
    # Main Training Loop
    # ---------------------------------------------------------------------------
    for epoch in range(args.stage1_max_epochs + args.stage2_epochs):
        total_epoch_count += 1

        # Train Amplitude
        model_amplitude.train(True)
        if current_stage == 2:
            filter_amplitude.train(True)
        total_loss_a_train = 0.0
        for p_in, a_target in a_train_loader:
            p_in, a_target = p_in.to(device), a_target.to(device)
            opt_amplitude.zero_grad()
            pred_wf_a = full_model_ampl(p_in)
            loss_a = loss_fn_amplitude(pred_wf_a, a_target)
            loss_a.backward()
            opt_amplitude.step()
            total_loss_a_train += loss_a.item()
        total_loss_a_train /= max(len(a_train_loader), 1)

        # Val Amplitude
        full_model_ampl.eval()
        total_loss_a_val = 0.0
        with torch.no_grad():
            for p_in, a_target in a_val_loader:
                p_in, a_target = p_in.to(device), a_target.to(device)
                pred_wf_a = full_model_ampl(p_in)
                total_loss_a_val += loss_fn_amplitude(pred_wf_a, a_target).item()
        total_loss_a_val /= max(len(a_val_loader), 1)
        scheduler_amplitude.step(total_loss_a_val)

        # Train Phase
        model_phase.train(True)
        if current_stage == 2:
            filter_phase.train(True)
        total_loss_p_train = 0.0
        for p_in, ph_target in p_train_loader:
            p_in, ph_target = p_in.to(device), ph_target.to(device)
            opt_phase.zero_grad()
            pred_wf_p = full_model_phase(p_in)
            loss_p = loss_fn_phase(pred_wf_p, ph_target)
            loss_p.backward()
            opt_phase.step()
            total_loss_p_train += loss_p.item()
        total_loss_p_train /= max(len(p_train_loader), 1)

        # Val Phase
        full_model_phase.eval()
        total_loss_p_val = 0.0
        with torch.no_grad():
            for p_in, ph_target in p_val_loader:
                p_in, ph_target = p_in.to(device), ph_target.to(device)
                pred_wf_p = full_model_phase(p_in)
                total_loss_p_val += loss_fn_phase(pred_wf_p, ph_target).item()
        total_loss_p_val /= max(len(p_val_loader), 1)
        scheduler_phase.step(total_loss_p_val)

        lr_a = scheduler_amplitude.get_last_lr()[0]
        lr_p = scheduler_phase.get_last_lr()[0]

        history["epoch"].append(total_epoch_count)
        history["stage"].append(current_stage)
        history["train_loss_amplitude"].append(total_loss_a_train)
        history["val_loss_amplitude"].append(total_loss_a_val)
        history["lr_amplitude"].append(lr_a)
        history["train_loss_phase"].append(total_loss_p_train)
        history["val_loss_phase"].append(total_loss_p_val)
        history["lr_phase"].append(lr_p)

        # Save Best Models
        if total_loss_a_val < best_vloss_amplitude:
            best_vloss_amplitude = total_loss_a_val
            torch.save(full_model_ampl.state_dict(), f"{title}_amplitude_best.pt")

        if total_loss_p_val < best_vloss_phase:
            best_vloss_phase = total_loss_p_val
            torch.save(full_model_phase.state_dict(), f"{title}_phase_best.pt")

        # Periodic Progress Output
        if total_epoch_count % 25 == 0 or total_epoch_count == 1:
            print(
                f"[Stage {current_stage} | Ep {total_epoch_count}] "
                f"Ampl: train={total_loss_a_train:.3e} val={total_loss_a_val:.3e} (best={best_vloss_amplitude:.3e}, lr={lr_a:.1e}) | "
                f"Phase: train={total_loss_p_train:.3e} val={total_loss_p_val:.3e} (best={best_vloss_phase:.3e}, lr={lr_p:.1e})"
            )

        # -----------------------------------------------------------------------
        # Check Transition from Stage 1 to Stage 2
        # -----------------------------------------------------------------------
        if current_stage == 1:
            trigger_activated = False
            trigger_reason = ""

            # Check LR drop threshold trigger
            if args.stage1_trigger in ["auto", "lr"]:
                if lr_a <= args.stage1_lr_threshold and lr_p <= args.stage1_lr_threshold:
                    trigger_activated = True
                    trigger_reason = f"Learning rate reached floor (Ampl LR={lr_a:.1e}, Phase LR={lr_p:.1e} <= {args.stage1_lr_threshold:.1e})"

            # Check Max Epochs trigger
            if not trigger_activated and (total_epoch_count >= args.stage1_max_epochs):
                trigger_activated = True
                trigger_reason = f"Reached Stage 1 max epochs limit ({args.stage1_max_epochs})"

            if trigger_activated:
                current_stage = 2
                stage1_completed_epochs = total_epoch_count
                stage2_start_epoch = total_epoch_count
                print(f"\n=======================================================")
                print(f"=== [TRIGGERING STAGE 2 at Epoch {total_epoch_count}] ===")
                print(f"  Reason: {trigger_reason}")
                print(f"  Unfreezing Decoders: Joint training with differential LR")
                print(f"  Mapper LR: {args.stage2_lr_mapper} | Decoder LR: {args.stage2_lr_decoder}")
                print(f"=======================================================\n")

                # Unfreeze Decoders
                full_model_ampl.decoder.requires_grad_(True)
                full_model_phase.decoder.requires_grad_(True)

                # Initialize Stage 2 Differential Optimizers
                opt_amplitude = opt_add.SOAP([
                    {"params": full_model_ampl.mapper.parameters(), "lr": args.stage2_lr_mapper},
                    {"params": full_model_ampl.decoder.parameters(), "lr": args.stage2_lr_decoder},
                ], betas=(0.95, 0.999), shampoo_beta=0.99, weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001)

                opt_phase = opt_add.SOAP([
                    {"params": full_model_phase.mapper.parameters(), "lr": args.stage2_lr_mapper},
                    {"params": full_model_phase.decoder.parameters(), "lr": args.stage2_lr_decoder},
                ], betas=(0.95, 0.999), shampoo_beta=0.99, weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001)

                scheduler_amplitude = lr_scheduler.ReduceLROnPlateau(
                    opt_amplitude, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-7, cooldown=25
                )
                scheduler_phase = lr_scheduler.ReduceLROnPlateau(
                    opt_phase, factor=0.5, patience=50, threshold=0.0001, threshold_mode="rel", min_lr=1e-7, cooldown=25
                )
                early_stop_amplitude.best_loss = best_vloss_amplitude
                early_stop_phase.best_loss = best_vloss_phase
                continue

        # -----------------------------------------------------------------------
        # Stage 2 Early Stopping & Max Epochs Check
        # -----------------------------------------------------------------------
        if current_stage == 2:
            stage2_epochs_done = total_epoch_count - stage2_start_epoch
            if args.early_stop > 0 and stage2_epochs_done >= 50:
                stop_a = early_stop_amplitude.step(total_loss_a_val, lr_a)
                stop_p = early_stop_phase.step(total_loss_p_val, lr_p)
                if stop_a and stop_p:
                    print(f"\n=== [Stage 2 Early Stopping Triggered at Epoch {total_epoch_count}] ===")
                    print(f"  Amplitude: {early_stop_amplitude.stop_reason}")
                    print(f"  Phase:     {early_stop_phase.stop_reason}")
                    break

            if stage2_epochs_done >= args.stage2_epochs:
                print(f"\n=== [Stage 2 Completed Max Allowed Epochs: {args.stage2_epochs}] ===")
                break

    # ---------------------------------------------------------------------------
    # Save Metrics CSV
    # ---------------------------------------------------------------------------
    csv_filename = f"{title}_metrics.csv"
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_filename, index=False)
    print(f"\nUnified metrics CSV saved: {csv_filename}")

    # ---------------------------------------------------------------------------
    # Plot Training Curves
    # ---------------------------------------------------------------------------
    plot_training_curves(
        epochs=history["epoch"],
        train_losses=history["train_loss_amplitude"],
        val_losses=history["val_loss_amplitude"],
        lrs=history["lr_amplitude"],
        run_title=f"{title} (Stage 2 at Ep {stage2_start_epoch})",
        metric_name="Amplitude",
        output_path=f"{title}_amplitude_plot.png",
    )
    plot_training_curves(
        epochs=history["epoch"],
        train_losses=history["train_loss_phase"],
        val_losses=history["val_loss_phase"],
        lrs=history["lr_phase"],
        run_title=f"{title} (Stage 2 at Ep {stage2_start_epoch})",
        metric_name="Phase",
        output_path=f"{title}_phase_plot.png",
    )

    # ---------------------------------------------------------------------------
    # Evaluate Waveform Mismatches on NRHyb Test Set
    # ---------------------------------------------------------------------------
    print(f"\nLoading best saved models for test set mismatch evaluation...")
    full_model_ampl.load_state_dict(torch.load(f"{title}_amplitude_best.pt", map_location=device))
    full_model_phase.load_state_dict(torch.load(f"{title}_phase_best.pt", map_location=device))
    full_model_ampl.eval()
    full_model_phase.eval()

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

    print(f"\n=======================================================")
    print(f"=== Final Test Set Mismatch Stats ({title}) ===")
    print(f"=======================================================")
    print(f"  Median Mismatch: {np.median(mismatches):.3e}")
    print(f"  Mean Mismatch:   {np.mean(mismatches):.3e}")
    print(f"  Max Mismatch:    {np.max(mismatches):.3e}")
    print(f"  Fraction < 1e-4: {100.0 * np.mean(mismatches < 1e-4):.2f}%")
    print(f"  Fraction < 1e-3: {100.0 * np.mean(mismatches < 1e-3):.2f}%")
    print(f"Saved artifacts:")
    print(f"  - Amplitude Model: {title}_amplitude_best.pt")
    print(f"  - Phase Model:     {title}_phase_best.pt")
    print(f"  - Metrics CSV:     {csv_filename}")
    print(f"  - Mismatch Plot:   {plot_mismatch_filename}")


if __name__ == "__main__":
    main()

