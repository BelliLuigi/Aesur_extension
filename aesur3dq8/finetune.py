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
from utils import (
    get_phase,
    get_NR_data,
    farthest_point_sampling,
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
    """Auto-detect mapper architecture from saved checkpoint state dict."""
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


def load_checkpoint_weights(model: FullModel, ckpt_path: str, device: torch.device):
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


def build_full_models(checkpoint_ampl: str, checkpoint_phase: str, mapper_name: str = "auto", device: torch.device = None):
    """Initializes FullModels and reloads pretrained mapper and decoder weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    load_checkpoint_weights(full_model_ampl, checkpoint_ampl, device)
    load_checkpoint_weights(full_model_phase, checkpoint_phase, device)
    return full_model_ampl, full_model_phase


def parse_args():
    parser = argparse.ArgumentParser(description="Single-Stage Joint Finetuning of FullModels on SXS Waveforms")
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
        default="model_weights/soapFull2628TMapper_resumed_amplitude_best.pt",
        help="Path to pretrained amplitude checkpoint (default: model_weights/soapFull2628TMapper_resumed_amplitude_best.pt)",
    )
    parser.add_argument(
        "--checkpoint_phase",
        type=str,
        default="model_weights/soapFull2628TMapper_resumed_phase_best.pt",
        help="Path to pretrained phase checkpoint (default: model_weights/soapFull2628TMapper_resumed_phase_best.pt)",
    )
    parser.add_argument("--folds", type=int, default=1, choices=[1, 5], help="Number of training folds: 1 (single-stage train/val split) or 5 (5-Fold CV) (default: 1)")
    parser.add_argument("--epochs", type=int, default=1500, help="Total training epochs (default: 1500)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and eval (default: 32)")
    parser.add_argument("--lr_mapper", type=float, default=1e-4, help="Learning rate for mapper (default: 1e-4)")
    parser.add_argument("--lr_decoder", type=float, default=1e-5, help="Learning rate for decoder (default: 1e-5, set 0 to freeze)")
    parser.add_argument("--early_stop", type=int, default=100, help="Early stopping patience in epochs (default: 100, 0 to disable)")
    return parser.parse_args()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        print(f"Allocated GPU: {prop.name} | Compute Capability: {prop.major}.{prop.minor} | VRAM: {prop.total_memory / (1024**3):.1f} GB")
    else:
        print("Using CPU device.")

    args = parse_args()

    title = args.title
    if not title:
        try:
            title = input("Enter title of this run: ").strip()
        except EOFError:
            title = ""
        if not title:
            title = f"Finetune_SOAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"=== Starting Single-Stage Finetuning: {title} | Folds: {args.folds} | Epochs: {args.epochs} | Mapper LR: {args.lr_mapper} | Decoder LR: {args.lr_decoder} ===")

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

    # Held-out 10% test set (~37 samples) via FPS
    n_test = int(0.10 * len(joint_dataset))
    test_idx = farthest_point_sampling(params_tensor, n_test)

    mask_pool = torch.ones(len(joint_dataset), dtype=torch.bool)
    mask_pool[test_idx] = False
    train_val_idx = torch.arange(len(joint_dataset))[mask_pool]

    test_loader = DataLoader(Subset(joint_dataset, test_idx), batch_size=args.batch_size, shuffle=False)
    print(f"Test set (held-out FPS): {len(test_idx)} samples | Training pool: {len(train_val_idx)} samples")

    # ---------------------------------------------------------------------------
    # Setup Folds: Single-Stage Split (folds=1) or K-Fold CV (folds=5)
    # ---------------------------------------------------------------------------
    k = args.folds
    perm = train_val_idx[torch.randperm(len(train_val_idx), generator=torch.Generator().manual_seed(42))]

    if k == 1:
        n_val = max(1, int(0.15 * len(perm)))
        fold_splits = [(perm[n_val:], perm[:n_val])]
    else:
        fold_chunks = torch.chunk(perm, chunks=k)
        fold_splits = [
            (torch.cat([fold_chunks[i] for i in range(k) if i != f]), fold_chunks[f])
            for f in range(k)
        ]

    cv_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        prefix = f"{title}_fold{fold_num}" if k > 1 else title

        print(f"\n==================================================")
        print(f"  Training Split {fold_num}/{k} | Prefix: {prefix}")
        print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        print(f"==================================================")

        train_loader = DataLoader(Subset(joint_dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(joint_dataset, val_idx),   batch_size=args.batch_size, shuffle=False)

        # Build FullModels
        full_model_ampl, full_model_phase = build_full_models(
            args.checkpoint_ampl, args.checkpoint_phase, args.mapper, device=device
        )

        loss_fn_amplitude = nn.SmoothL1Loss(beta=1e-3)
        loss_fn_phase = nn.SmoothL1Loss(beta=1e-3)

        # Configure Single-Stage Joint Optimizer
        full_model_ampl.mapper.requires_grad_(True)
        full_model_phase.mapper.requires_grad_(True)

        if args.lr_decoder > 0:
            full_model_ampl.decoder.requires_grad_(True)
            full_model_phase.decoder.requires_grad_(True)
            params_a = [
                {"params": full_model_ampl.mapper.parameters(), "lr": args.lr_mapper},
                {"params": full_model_ampl.decoder.parameters(), "lr": args.lr_decoder},
            ]
            params_p = [
                {"params": full_model_phase.mapper.parameters(), "lr": args.lr_mapper},
                {"params": full_model_phase.decoder.parameters(), "lr": args.lr_decoder},
            ]
            print(f"Joint Training: Mapper LR={args.lr_mapper}, Decoder LR={args.lr_decoder}")
        else:
            full_model_ampl.decoder.requires_grad_(False)
            full_model_phase.decoder.requires_grad_(False)
            params_a = [{"params": full_model_ampl.mapper.parameters(), "lr": args.lr_mapper}]
            params_p = [{"params": full_model_phase.mapper.parameters(), "lr": args.lr_mapper}]
            print(f"Decoder Frozen: Optimizing Mapper Only (LR={args.lr_mapper})")

        opt_fn_amplitude = opt_add.SOAP(
            params_a, betas=(0.95, 0.999), shampoo_beta=0.99, weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
        )
        opt_fn_phase = opt_add.SOAP(
            params_p, betas=(0.95, 0.999), shampoo_beta=0.99, weight_decay=0, precondition_frequency=25, merge_dims=False, eps=0.0001
        )

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

        # Training Loop
        for epoch in range(args.epochs):
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

            # Validation step
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

            scheduler_amplitude.step(total_loss_amplitudeV)
            scheduler_phase.step(total_loss_phaseV)

            current_lr_ampl = opt_fn_amplitude.param_groups[0]["lr"]
            current_lr_ph = opt_fn_phase.param_groups[0]["lr"]

            history["epoch"].append(epoch + 1)
            history["train_loss_amplitude"].append(total_loss_amplitudeT)
            history["val_loss_amplitude"].append(total_loss_amplitudeV)
            history["lr_amplitude"].append(current_lr_ampl)
            history["train_loss_phase"].append(total_loss_phaseT)
            history["val_loss_phase"].append(total_loss_phaseV)
            history["lr_phase"].append(current_lr_ph)

            if total_loss_amplitudeV < best_vloss_amplitude:
                best_vloss_amplitude = total_loss_amplitudeV
                torch.save(full_model_ampl.state_dict(), f"{prefix}_amplitude_best.pt")

            if total_loss_phaseV < best_vloss_phase:
                best_vloss_phase = total_loss_phaseV
                torch.save(full_model_phase.state_dict(), f"{prefix}_phase_best.pt")

            if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
                print(
                    f"[{prefix}] Epoch [{epoch + 1}/{args.epochs}] "
                    f"Ampl: train={total_loss_amplitudeT:.3e} val={total_loss_amplitudeV:.3e} (best={best_vloss_amplitude:.3e}) | "
                    f"Phase: train={total_loss_phaseT:.3e} val={total_loss_phaseV:.3e} (best={best_vloss_phase:.3e})"
                )

            if args.early_stop > 0 and epoch >= 50:
                stop_a = early_stop_amplitude.step(total_loss_amplitudeV, current_lr_ampl)
                stop_p = early_stop_phase.step(total_loss_phaseV, current_lr_ph)
                if stop_a and stop_p:
                    print(f"[{prefix}] Early Stopping Triggered at Epoch {epoch + 1}")
                    break

        # Save history & curves
        df_history = pd.DataFrame(history)
        df_history.to_csv(f"{prefix}_metrics.csv", index=False)

        plot_training_curves(
            epochs=history["epoch"],
            train_losses=history["train_loss_amplitude"],
            val_losses=history["val_loss_amplitude"],
            lrs=history["lr_amplitude"],
            run_title=prefix,
            metric_name="Amplitude",
            output_path=f"{prefix}_amplitude_plot.png",
        )
        plot_training_curves(
            epochs=history["epoch"],
            train_losses=history["train_loss_phase"],
            val_losses=history["val_loss_phase"],
            lrs=history["lr_phase"],
            run_title=prefix,
            metric_name="Phase",
            output_path=f"{prefix}_phase_plot.png",
        )

        cv_summaries.append({
            "fold": fold_num,
            "best_val_loss_ampl": best_vloss_amplitude,
            "best_val_loss_phase": best_vloss_phase,
            "epochs_trained": len(history["epoch"]),
        })

    # ---------------------------------------------------------------------------
    # Evaluation on Held-Out Test Set (37 Waveforms)
    # ---------------------------------------------------------------------------
    df_cv = pd.DataFrame(cv_summaries)
    if k > 1:
        df_cv.to_csv(f"{title}_cv_summary.csv", index=False)
        print("\n=== Cross-Validation Results across 5 Folds ===")
        print(df_cv.to_string(index=False))
        best_fold_idx = int(np.argmin(df_cv["best_val_loss_ampl"] + df_cv["best_val_loss_phase"])) + 1
        best_ampl_path = f"{title}_fold{best_fold_idx}_amplitude_best.pt"
        best_phase_path = f"{title}_fold{best_fold_idx}_phase_best.pt"

        # Alias best fold to canonical output names
        final_ampl_path = f"{title}_amplitude_best.pt"
        final_phase_path = f"{title}_phase_best.pt"
        torch.save(torch.load(best_fold_ampl_path, map_location=device), final_ampl_path)
        torch.save(torch.load(best_fold_phase_path, map_location=device), final_phase_path)

        # Cleanup fold checkpoints
        for f in range(1, k + 1):
            for suffix in ["_amplitude_best.pt", "_phase_best.pt"]:
                p = f"{title}_fold{f}{suffix}"
                if os.path.exists(p):
                    os.remove(p)
    else:
        final_ampl_path = f"{title}_amplitude_best.pt"
        final_phase_path = f"{title}_phase_best.pt"

    print(f"\nEvaluating final model on held-out test set ({len(test_idx)} waveforms)...")
    eval_model_ampl, eval_model_phase = build_full_models(
        args.checkpoint_ampl, args.checkpoint_phase, args.mapper, device=device
    )
    eval_model_ampl.load_state_dict(torch.load(final_ampl_path, map_location=device))
    eval_model_phase.load_state_dict(torch.load(final_phase_path, map_location=device))
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
        run_title=title,
        output_path=plot_mismatch_filename,
    )

    print(f"\n=== Final Held-Out Test Set Mismatch Stats ({title}) ===")
    print(f"  Median: {np.median(mismatches):.2e} | Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}")
    print(f"Artifacts saved:")
    print(f"  - Best Amplitude Model: {final_ampl_path}")
    print(f"  - Best Phase Model:     {final_phase_path}")
    print(f"  - Test Mismatch Plot:   {plot_mismatch_filename}")


if __name__ == "__main__":
    main()
