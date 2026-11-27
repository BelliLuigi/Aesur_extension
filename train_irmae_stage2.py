#!/usr/bin/env python3
"""
IRMAE Stage 2: Parameter-to-Waveform training from a Stage 1 IRMAE checkpoint.

Runs two sequential phases in float64 (SOAP + CosineAnnealingLR + torch.compile):
  Phase 2a: Full(NarrowDecoder frozen) — trains mapper only.
  Phase 2b: Full(NarrowDecoder unfrozen) — trains mapper + decoder jointly per branch.

Losses (from utils_mlp.py):
  - Amplitude: RelativeAmplitudeRMSLoss
  - LogFreq:   AmplitudeWeightedPhaseLoss (A^2-weighted cumulative phase RMS + L1 stabilizer)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models_mlp import Full, LinearMatrices, NarrowDecoder, NarrowEncoder, irmae
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
    p = argparse.ArgumentParser(description="IRMAE Stage 2: Mapper+Frozen Dec -> Mapper+Unfrozen Dec")
    p.add_argument("--stage1_ckpt", type=str, required=True, help="Path to Stage 1 IRMAE checkpoint (.pt)")
    p.add_argument("--title", type=str, default=None, help="Run prefix (default: Stage 1 title)")
    p.add_argument("--dataset", type=str, default=None, help="Dataset HDF5 (default: from Stage 1 ckpt)")
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--full_neurons", type=int, default=128, help="Full mapper neurons_unit (default: 128)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=2048)

    p.add_argument("--frozen_epochs", type=int, default=3000, help="Phase 2a frozen-decoder epochs")
    p.add_argument("--lr_frozen", type=float, default=1e-3)
    p.add_argument("--min_lr_frozen", type=float, default=1e-7)

    p.add_argument("--unfrozen_epochs", type=int, default=3000, help="Phase 2b unfrozen-decoder epochs")
    p.add_argument("--lr_unfrozen", type=float, default=1e-4)
    p.add_argument("--min_lr_unfrozen", type=float, default=1e-7)

    p.add_argument("--lf_l1_weight", type=float, default=None, help="Override L1 stabilizer weight (default: from S1 ckpt)")
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

    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    title = args.title or ckpt["title"]
    dataset_path = args.dataset or ckpt["dataset"]
    latent_dim = int(ckpt["latent_dim"])
    ae_neurons = int(ckpt["ae_neurons"])
    lm_layers = int(ckpt["lm_layers"])
    omega_floor = float(ckpt.get("omega_floor", 1e-6))
    lf_l1_weight = float(args.lf_l1_weight if args.lf_l1_weight is not None else ckpt.get("lf_l1_weight", 0.05))
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

    # Reconstruct Stage 1 IRMAE models to extract NarrowDecoders + baseline Stage 1 eval (for R^2 alignment plots)
    ae_amp = irmae(
        NarrowEncoder(sample_len, latent_dim, n=ae_neurons),
        LinearMatrices(latent_dim, lm_layers),
        NarrowDecoder(sample_len, latent_dim, n=ae_neurons),
    ).to(device=device, dtype=dtype)
    ae_lf = irmae(
        NarrowEncoder(sample_len, latent_dim, n=ae_neurons),
        LinearMatrices(latent_dim, lm_layers),
        NarrowDecoder(sample_len, latent_dim, n=ae_neurons),
    ).to(device=device, dtype=dtype)
    ae_amp.load_state_dict(ckpt["ae_amp"])
    ae_lf.load_state_dict(ckpt["ae_lf"])

    diag_s1 = evaluate_stage(
        ae_amp, ae_lf, nn.Sequential(ae_amp.e, ae_amp.lm), nn.Sequential(ae_lf.e, ae_lf.lm),
        ampl_t[test_idx], logfreq_t[test_idx],
        data["h_test_true"], data["true_phase_test"], data["true_log_om_test"],
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    print_mismatch_stats("Stage 1: AE (loaded)", diag_s1)

    loss_amp_fn = nn.L1Loss()
    loss_lf_fn = AmplitudeWeightedPhaseLoss(mu_lf_t, sd_lf_t, l1_weight=lf_l1_weight).to(device=device)

    pin = torch.cuda.is_available()
    fa_tr = DataLoader(TensorDataset(params_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    fa_va = DataLoader(TensorDataset(params_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)
    fl_tr = DataLoader(TensorDataset(params_t[train_idx], logfreq_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    fl_va = DataLoader(TensorDataset(params_t[val_idx], logfreq_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    meta = {
        "title": title,
        "arch": "irmae",
        "dataset": dataset_path,
        "sample_len": sample_len,
        "labels": n_labels,
        "latent_dim": latent_dim,
        "ae_neurons": ae_neurons,
        "lm_layers": lm_layers,
        "full_neurons": args.full_neurons,
        "f64": True,
        "omega_floor": omega_floor,
        "lf_l1_weight": lf_l1_weight,
        "norm_path": norm_path,
        "stage1_ckpt": args.stage1_ckpt,
    }
    stage_diags = {"Stage 1: AE": diag_s1}
    histories = []
    pdf_path = f"runs/{title}_stage2_plots.pdf"

    # ------------------------------------------------------------------
    # Phase 2a: Mapper + Frozen Decoder
    # ------------------------------------------------------------------
    full_amp = Full(ae_amp.d, labels=n_labels, latent_dim=latent_dim, neurons_unit=args.full_neurons, trainable=False).to(device=device, dtype=dtype)
    full_lf = Full(ae_lf.d, labels=n_labels, latent_dim=latent_dim, neurons_unit=args.full_neurons, trainable=False).to(device=device, dtype=dtype)

    exec_2a_amp = compile_model(full_amp, "IRMAE_S2a_Frozen_Amp")
    exec_2a_lf = compile_model(full_lf, "IRMAE_S2a_Frozen_LF")

    s2a_a_tr, s2a_a_va, s2a_a_lr, s2a_a_best = run_branch_stage(
        "Stage 2a [Full Amplitude | Frozen Dec]", exec_2a_amp, full_amp, list(full_amp.mapper.parameters()),
        fa_tr, fa_va, loss_amp_fn, args.frozen_epochs, args.lr_frozen, args.min_lr_frozen, device,
    )
    s2a_l_tr, s2a_l_va, s2a_l_lr, s2a_l_best = run_branch_stage(
        "Stage 2a [Full LogFreq | Frozen Dec]", exec_2a_lf, full_lf, list(full_lf.mapper.parameters()),
        fl_tr, fl_va, loss_lf_fn, args.frozen_epochs, args.lr_frozen, args.min_lr_frozen, device,
    )

    if s2a_a_best is not None:
        full_amp.load_state_dict(s2a_a_best)
    if s2a_l_best is not None:
        full_lf.load_state_dict(s2a_l_best)

    diag_2a = evaluate_stage(
        full_amp, full_lf, full_amp.mapper, full_lf.mapper,
        params_t[test_idx], params_t[test_idx],
        data["h_test_true"], data["true_phase_test"], data["true_log_om_test"],
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    stage_diags["Stage 2: Full (Frozen Dec)"] = diag_2a
    histories.extend([
        ("Stage 2a: Full Amp (Frozen)", s2a_a_tr, s2a_a_va, s2a_a_lr),
        ("Stage 2a: Full LogFreq (Frozen)", s2a_l_tr, s2a_l_va, s2a_l_lr),
    ])
    print_mismatch_stats("Stage 2a: Full (Frozen Dec)", diag_2a)
    frozen_ckpt = f"model_weights/{title}_stage2_frozen_best.pt"
    torch.save({**meta, "full_amp": full_amp.state_dict(), "full_lf": full_lf.state_dict()}, frozen_ckpt)
    save_report_pdf(pdf_path, stage_diags, histories, f"{title} (IRMAE Stage 2)")

    # ------------------------------------------------------------------
    # Phase 2b: Mapper + Unfrozen Decoder (per-branch)
    # ------------------------------------------------------------------
    for p in full_amp.parameters():
        p.requires_grad_(True)
    for p in full_lf.parameters():
        p.requires_grad_(True)

    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    exec_2b_amp = compile_model(full_amp, "IRMAE_S2b_Unfrozen_Amp")
    exec_2b_lf = compile_model(full_lf, "IRMAE_S2b_Unfrozen_LF")

    s2b_a_tr, s2b_a_va, s2b_a_lr, s2b_a_best = run_branch_stage(
        "Stage 2b [Full Amplitude | Unfrozen Dec]", exec_2b_amp, full_amp, list(full_amp.parameters()),
        fa_tr, fa_va, loss_amp_fn, args.unfrozen_epochs, args.lr_unfrozen, args.min_lr_unfrozen, device,
    )
    s2b_l_tr, s2b_l_va, s2b_l_lr, s2b_l_best = run_branch_stage(
        "Stage 2b [Full LogFreq | Unfrozen Dec]", exec_2b_lf, full_lf, list(full_lf.parameters()),
        fl_tr, fl_va, loss_lf_fn, args.unfrozen_epochs, args.lr_unfrozen, args.min_lr_unfrozen, device,
    )

    if s2b_a_best is not None:
        full_amp.load_state_dict(s2b_a_best)
    if s2b_l_best is not None:
        full_lf.load_state_dict(s2b_l_best)

    diag_2b = evaluate_stage(
        full_amp, full_lf, full_amp.mapper, full_lf.mapper,
        params_t[test_idx], params_t[test_idx],
        data["h_test_true"], data["true_phase_test"], data["true_log_om_test"],
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    stage_diags["Stage 3: Full (Trainable Dec)"] = diag_2b
    histories.extend([
        ("Stage 2b: Full Amp (Unfrozen)", s2b_a_tr, s2b_a_va, s2b_a_lr),
        ("Stage 2b: Full LogFreq (Unfrozen)", s2b_l_tr, s2b_l_va, s2b_l_lr),
    ])
    print_mismatch_stats("Stage 2b: Full (Unfrozen Dec)", diag_2b)

    unfrozen_ckpt = f"model_weights/{title}_stage2_unfrozen_best.pt"
    npz_path = f"runs/{title}_stage2_mismatches.npz"
    csv_path = f"runs/{title}_stage2_metrics.csv"

    torch.save({**meta, "full_amp": full_amp.state_dict(), "full_lf": full_lf.state_dict()}, unfrozen_ckpt)
    save_report_pdf(pdf_path, stage_diags, histories, f"{title} (IRMAE Stage 2)")
    np.savez_compressed(npz_path, stage1_ae=diag_s1["mm"], stage2_frozen=diag_2a["mm"], stage2_unfrozen=diag_2b["mm"])

    rows = []
    for tag, (tr, va, lr) in [
        ("stage2a_amp_frozen", (s2a_a_tr, s2a_a_va, s2a_a_lr)),
        ("stage2a_lf_frozen", (s2a_l_tr, s2a_l_va, s2a_l_lr)),
        ("stage2b_amp_unfrozen", (s2b_a_tr, s2b_a_va, s2b_a_lr)),
        ("stage2b_lf_unfrozen", (s2b_l_tr, s2b_l_va, s2b_l_lr)),
    ]:
        for ep_i, (t_v, v_v, l_v) in enumerate(zip(tr, va, lr), 1):
            rows.append({"stage": tag, "epoch": ep_i, "train_loss": t_v, "val_loss": v_v, "lr": l_v})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nSaved Stage 2 artifacts:")
    print(f"  - {frozen_ckpt}")
    print(f"  - {unfrozen_ckpt}")
    print(f"  - {pdf_path}")
    print(f"  - {npz_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
