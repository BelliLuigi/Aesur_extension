#!/usr/bin/env python3
"""
IRMAE Stage 1: Train Amplitude & Log-Frequency IRMAE Autoencoders in float64.

Architecture (from models_mlp.py):
  NarrowEncoder -> LinearMatrices (orthogonal init) -> NarrowDecoder wrapped in `irmae`.
Losses (from utils_mlp.py):
  - Amplitude AE: RelativeAmplitudeRMSLoss
  - LogFreq AE:   AmplitudeWeightedPhaseLoss (A^2-weighted cumulative phase RMS + L1 stabilizer)
Optimizer: SOAP + CosineAnnealingLR + torch.compile (float64).
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models_mlp import LinearMatrices, NarrowDecoder, NarrowEncoder, irmae
from utils_mlp import (
    AmplitudeWeightedPhaseLoss,
    RelativeAmplitudeRMSLoss,
    compile_model,
    configure_precision,
    evaluate_stage,
    load_waveform_dataset,
    print_mismatch_stats,
    run_branch_stage,
    save_report_pdf,
)


def parse_args():
    p = argparse.ArgumentParser(description="IRMAE Stage 1: Amplitude + LogFreq Autoencoders (f64, SOAP, Cosine)")
    p.add_argument("--title", type=str, default=None, help="Run prefix for checkpoints and plots")
    p.add_argument("--dataset", type=str, default="datasets/NRSur7dq4v2_dataset.hdf")
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--latent_dim", type=int, default=24, help="Latent dimension (default: 24)")
    p.add_argument("--ae_neurons", type=int, default=128, help="NarrowEncoder/NarrowDecoder n (default: 128)")
    p.add_argument("--lm_layers", type=int, default=4, help="Number of LinearMatrices layers (default: 4)")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=2048)
    p.add_argument("--ae_epochs", type=int, default=3000)
    p.add_argument("--lr_ae", type=float, default=1e-3)
    p.add_argument("--min_lr_ae", type=float, default=1e-7)

    p.add_argument("--omega_floor", type=float, default=1e-6)
    p.add_argument("--eps", type=float, default=1e-30)
    p.add_argument("--lf_l1_weight", type=float, default=0.05)
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

    title = args.title or f"IRMAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    norm_path = f"model_weights/{title}_norm_params.npz"

    data = load_waveform_dataset(
        args.dataset,
        device=device,
        omega_floor=args.omega_floor,
        max_samples=args.max_samples,
        save_norm_path=norm_path,
        eps=args.eps,
    )
    sample_len = data["sample_len"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    ampl_t, logfreq_t = data["ampl_t"], data["logfreq_t"]
    mu_lf_t, sd_lf_t = data["mu_lf_t"], data["sd_lf_t"]

    # Build IRMAE autoencoders
    ae_amp = irmae(
        NarrowEncoder(sample_len, args.latent_dim, n=args.ae_neurons),
        LinearMatrices(args.latent_dim, args.lm_layers),
        NarrowDecoder(sample_len, args.latent_dim, n=args.ae_neurons),
    ).to(device=device, dtype=dtype)
    ae_lf = irmae(
        NarrowEncoder(sample_len, args.latent_dim, n=args.ae_neurons),
        LinearMatrices(args.latent_dim, args.lm_layers),
        NarrowDecoder(sample_len, args.latent_dim, n=args.ae_neurons),
    ).to(device=device, dtype=dtype)

    loss_amp_fn = nn.L1Loss()
    loss_lf_fn = AmplitudeWeightedPhaseLoss(mu_lf_t, sd_lf_t, l1_weight=args.lf_l1_weight).to(device=device)

    pin = torch.cuda.is_available()
    a_tr = DataLoader(TensorDataset(ampl_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    a_va = DataLoader(TensorDataset(ampl_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)
    l_tr = DataLoader(TensorDataset(logfreq_t[train_idx], logfreq_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    l_va = DataLoader(TensorDataset(logfreq_t[val_idx], logfreq_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    exec_ae_amp = compile_model(ae_amp, "IRMAE_Amplitude")
    exec_ae_lf = compile_model(ae_lf, "IRMAE_LogFreq")

    s1_a_tr, s1_a_va, s1_a_lr, s1_a_best = run_branch_stage(
        "Stage 1 [IRMAE Amplitude AE]", exec_ae_amp, ae_amp, list(ae_amp.parameters()),
        a_tr, a_va, loss_amp_fn, args.ae_epochs, args.lr_ae, args.min_lr_ae, device,
    )
    s1_l_tr, s1_l_va, s1_l_lr, s1_l_best = run_branch_stage(
        "Stage 1 [IRMAE LogFreq AE]", exec_ae_lf, ae_lf, list(ae_lf.parameters()),
        l_tr, l_va, loss_lf_fn, args.ae_epochs, args.lr_ae, args.min_lr_ae, device,
    )

    if s1_a_best is not None:
        ae_amp.load_state_dict(s1_a_best)
    if s1_l_best is not None:
        ae_lf.load_state_dict(s1_l_best)

    # Latent representation in IRMAE is lm(e(x)) (actual input to NarrowDecoder)
    lat_amp_fn = nn.Sequential(ae_amp.e, ae_amp.lm)
    lat_lf_fn = nn.Sequential(ae_lf.e, ae_lf.lm)

    diag_s1 = evaluate_stage(
        ae_amp, ae_lf, lat_amp_fn, lat_lf_fn,
        ampl_t[test_idx], logfreq_t[test_idx],
        data["h_test_true"], data["true_phase_test"], data["true_log_om_test"],
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    print_mismatch_stats("Stage 1: AE", diag_s1)

    ckpt_path = f"model_weights/{title}_stage1_irmae_best.pt"
    pdf_path = f"runs/{title}_stage1_plots.pdf"
    npz_path = f"runs/{title}_stage1_mismatches.npz"
    csv_path = f"runs/{title}_stage1_metrics.csv"

    meta = {
        "title": title,
        "arch": "irmae",
        "dataset": args.dataset,
        "sample_len": sample_len,
        "labels": data["n_labels"],
        "latent_dim": args.latent_dim,
        "ae_neurons": args.ae_neurons,
        "lm_layers": args.lm_layers,
        "f64": True,
        "omega_floor": args.omega_floor,
        "lf_l1_weight": args.lf_l1_weight,
        "norm_path": norm_path,
    }
    torch.save({**meta, "ae_amp": ae_amp.state_dict(), "ae_lf": ae_lf.state_dict()}, ckpt_path)

    histories = [
        ("Stage 1: IRMAE Amplitude AE", s1_a_tr, s1_a_va, s1_a_lr),
        ("Stage 1: IRMAE LogFreq AE", s1_l_tr, s1_l_va, s1_l_lr),
    ]
    save_report_pdf(pdf_path, {"Stage 1: AE": diag_s1}, histories, f"{title} (IRMAE Stage 1)")
    np.savez_compressed(npz_path, stage1_ae=diag_s1["mm"])

    rows = []
    for tag, (tr, va, lr) in [("stage1_amp_ae", (s1_a_tr, s1_a_va, s1_a_lr)), ("stage1_lf_ae", (s1_l_tr, s1_l_va, s1_l_lr))]:
        for ep_i, (t_v, v_v, l_v) in enumerate(zip(tr, va, lr), 1):
            rows.append({"stage": tag, "epoch": ep_i, "train_loss": t_v, "val_loss": v_v, "lr": l_v})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nSaved Stage 1 artifacts:")
    print(f"  - {norm_path}")
    print(f"  - {ckpt_path}")
    print(f"  - {pdf_path}")
    print(f"  - {npz_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
