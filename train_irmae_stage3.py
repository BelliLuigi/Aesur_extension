#!/usr/bin/env python3
"""
IRMAE Stage 3: Joint Waveform Fine-Tuning with direct LogMismatchLoss in float64.

Loads a Stage 2 IRMAE checkpoint (frozen or unfrozen), unfreezes both Amplitude and
Log-Frequency Full models (mapper + NarrowDecoder), and optimizes them jointly with
LogMismatchLoss = mean(log(1 - TasosOverlapTorch(h_true, h_pred) + log_mm_eps)).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models_mlp import Full, NarrowDecoder
from utils_mlp import (
    LogMismatchLoss,
    compile_model,
    configure_precision,
    evaluate_stage,
    load_waveform_dataset,
    print_mismatch_stats,
    run_joint_stage,
    save_report_pdf,
)


def parse_args():
    p = argparse.ArgumentParser(description="IRMAE Stage 3: Joint LogMismatchLoss Fine-Tuning (f64, SOAP, Cosine)")
    p.add_argument("--stage2_ckpt", type=str, required=True, help="Path to Stage 2 checkpoint (.pt)")
    p.add_argument("--title", type=str, default=None, help="Run prefix (default: Stage 2 title)")
    p.add_argument("--dataset", type=str, default=None, help="Dataset HDF5 (default: from Stage 2 ckpt)")
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=2048)
    p.add_argument("--finetune_epochs", type=int, default=1500)
    p.add_argument("--lr_finetune", type=float, default=5e-5)
    p.add_argument("--min_lr_finetune", type=float, default=5e-8)

    p.add_argument("--log_mm_eps", type=float, default=1e-8)
    p.add_argument("--eps", type=float, default=1e-30)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    configure_precision()

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        print(f"GPU: {prop.name} ({prop.major}.{prop.minor}) | VRAM: {prop.total_memory / (1024**3):.1f} GB")
    else:
        print("Device: CPU")

    ckpt = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    title = args.title or ckpt["title"]
    dataset_path = args.dataset or ckpt["dataset"]
    latent_dim = int(ckpt["latent_dim"])
    ae_neurons = int(ckpt["ae_neurons"])
    full_neurons = int(ckpt["full_neurons"])
    omega_floor = float(ckpt.get("omega_floor", 1e-6))
    norm_path = ckpt.get("norm_path")

    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    data = load_waveform_dataset(
        dataset_path,
        device=device,
        omega_floor=omega_floor,
        max_samples=args.max_samples,
        norm_path=norm_path,
        eps=args.eps,
    )
    sample_len = data["sample_len"]
    n_labels = data["n_labels"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    params_t, ampl_t, logfreq_t = data["params_t"], data["ampl_t"], data["logfreq_t"]
    mu_lf_t, sd_lf_t = data["mu_lf_t"], data["sd_lf_t"]

    full_amp = Full(
        NarrowDecoder(sample_len, latent_dim, n=ae_neurons),
        labels=n_labels, latent_dim=latent_dim, neurons_unit=full_neurons, trainable=True,
    ).to(device=device, dtype=dtype)
    full_lf = Full(
        NarrowDecoder(sample_len, latent_dim, n=ae_neurons),
        labels=n_labels, latent_dim=latent_dim, neurons_unit=full_neurons, trainable=True,
    ).to(device=device, dtype=dtype)
    full_amp.load_state_dict(ckpt["full_amp"])
    full_lf.load_state_dict(ckpt["full_lf"])

    # Baseline evaluation before Stage 3 fine-tuning
    diag_pre = evaluate_stage(
        full_amp, full_lf, full_amp.mapper, full_lf.mapper,
        params_t[test_idx], params_t[test_idx],
        data["h_test_true"], data["true_phase_test"], data["true_log_om_test"],
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    print_mismatch_stats("Pre-Finetune (Stage 2)", diag_pre)

    loss_joint_fn = LogMismatchLoss(mu_lf_t, sd_lf_t, log_eps=args.log_mm_eps, overlap_eps=args.eps).to(device=device)

    pin = torch.cuda.is_available()
    joint_tr = DataLoader(TensorDataset(params_t[train_idx], ampl_t[train_idx], logfreq_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    joint_va = DataLoader(TensorDataset(params_t[val_idx], ampl_t[val_idx], logfreq_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    exec_amp = compile_model(full_amp, "IRMAE_S3_Finetune_Amp")
    exec_lf = compile_model(full_lf, "IRMAE_S3_Finetune_LF")

    s3_tr, s3_va, s3_lr, s3_a_best, s3_l_best = run_joint_stage(
        "Stage 3 [IRMAE Joint Finetune | LogMismatch]",
        exec_amp, exec_lf, full_amp, full_lf,
        joint_tr, joint_va, loss_joint_fn,
        args.finetune_epochs, args.lr_finetune, args.min_lr_finetune, device,
    )

    if s3_a_best is not None:
        full_amp.load_state_dict(s3_a_best)
    if s3_l_best is not None:
        full_lf.load_state_dict(s3_l_best)

    diag_post = evaluate_stage(
        full_amp, full_lf, full_amp.mapper, full_lf.mapper,
        params_t[test_idx], params_t[test_idx],
        data["h_test_true"], data["true_phase_test"], data["true_log_om_test"],
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    print_mismatch_stats("Stage 3: Finetuned", diag_post)

    ckpt_path = f"model_weights/{title}_stage3_finetune_best.pt"
    pdf_path = f"runs/{title}_stage3_plots.pdf"
    npz_path = f"runs/{title}_stage3_mismatches.npz"
    csv_path = f"runs/{title}_stage3_metrics.csv"

    meta = {
        "title": title,
        "arch": "irmae",
        "dataset": dataset_path,
        "sample_len": sample_len,
        "labels": n_labels,
        "latent_dim": latent_dim,
        "ae_neurons": ae_neurons,
        "full_neurons": full_neurons,
        "f64": True,
        "omega_floor": omega_floor,
        "log_mm_eps": args.log_mm_eps,
        "norm_path": norm_path,
        "stage2_ckpt": args.stage2_ckpt,
    }
    torch.save({**meta, "full_amp": full_amp.state_dict(), "full_lf": full_lf.state_dict()}, ckpt_path)

    stage_diags = {
        "Stage 2: Full (Frozen Dec)": diag_pre,
        "Stage 3: Full (Trainable Dec)": diag_post,
    }
    histories = [("Stage 3: Joint Finetune (LogMismatch)", s3_tr, s3_va, s3_lr)]
    save_report_pdf(pdf_path, stage_diags, histories, f"{title} (IRMAE Stage 3 Finetune)")
    np.savez_compressed(npz_path, pre_finetune=diag_pre["mm"], stage3_finetuned=diag_post["mm"])

    rows = [
        {"stage": "stage3_joint_finetune", "epoch": ep_i, "train_loss": t_v, "val_loss": v_v, "lr": l_v}
        for ep_i, (t_v, v_v, l_v) in enumerate(zip(s3_tr, s3_va, s3_lr), 1)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nSaved Stage 3 artifacts:")
    print(f"  - {ckpt_path}")
    print(f"  - {pdf_path}")
    print(f"  - {npz_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
