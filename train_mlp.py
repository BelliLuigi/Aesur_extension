#!/usr/bin/env python3
"""
3-stage PyTorch training pipeline in float64 using models_mlp (Encoder, Decoder, Full),
SOAP optimizer, CosineAnnealingLR (no early stopping), and torch.compile:

  Targets: Amplitude and standardized Log-Frequency (train7ae_logfreq / TasosAE_LOGFREQ style).
  Stage 1: Amplitude AE + Log-Frequency AE (3000 epochs each, CosineAnnealingLR).
  Stage 2: Full(decoder frozen) for Amplitude + Log-Frequency (3000 epochs each).
  Stage 3: Full(decoder trainable) for Amplitude + Log-Frequency (3000 epochs each).
  Eval:    Computes and plots test-set waveform mismatch (TasosOverlapTorch) after EACH stage,
           plus a combined AE vs Full comparison plot.
"""

import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models_mlp import Decoder, Encoder, Full
from utils import TasosOverlapTorch
from utils_mlp import (
    AmplitudeWeightedPhaseLoss,
    Autoencoder,
    LogMismatchLoss,
    RelativeAmplitudeRMSLoss,
    compile_model,
    configure_precision,
    evaluate_stage,
    parse_args,
    phase_to_logfreq,
    print_mismatch_stats,
    reconstruct_waveform_torch,
    run_branch_stage,
    run_joint_stage,
    save_report_pdf,
    wave_to_phase,
)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    configure_precision()

    dtype = torch.float64
    np_dtype = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        print(f"GPU: {prop.name} ({prop.major}.{prop.minor}) | VRAM: {prop.total_memory / (1024**3):.1f} GB")
    else:
        print("Device: CPU")

    title = args.title or f"MLP_AmpLogFreq_SOAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = args.dataset
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'")

    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # 1. Load waveforms & parameters
    print(f"Loading '{filepath}'...")
    with h5py.File(filepath, "r") as f:
        wf_ds = f["Waveforms"]
        params_ds = f["Parameters"]
        total_avail, sample_len = wf_ds.shape
        n_samples = min(total_avail, args.max_samples) if args.max_samples else total_avail
        n_labels = params_ds.shape[1]

        params_np = params_ds[:n_samples].astype(np_dtype)
        ampl_np = np.empty((n_samples, sample_len), dtype=np_dtype)
        logfreq_np = np.empty((n_samples, sample_len), dtype=np_dtype)

        chunk_size = 10000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            w = wf_ds[start:end]
            ampl_np[start:end] = np.abs(w).astype(np_dtype)
            ph = wave_to_phase(w)
            logfreq_np[start:end] = phase_to_logfreq(ph, omega_floor=args.omega_floor, dtype=np_dtype)

    print(f"Loaded {n_samples} waveforms (len={sample_len}, labels={n_labels}).")

    # 2. Split 80/10/10 identically to other scripts
    idx_all = np.arange(n_samples)
    gen = torch.Generator().manual_seed(42)
    tr_sub, va_sub, te_sub = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)
    train_idx = np.array(tr_sub.indices)
    val_idx = np.array(va_sub.indices)
    test_idx = np.array(te_sub.indices)
    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 3. Train-only normalization for log-frequency (per timestep) and 7D parameters
    mu_lf = np.zeros(sample_len, dtype=np.float64)
    sq_lf = np.zeros(sample_len, dtype=np.float64)
    for s in range(0, len(train_idx), 5000):
        blk = logfreq_np[train_idx[s:s + 5000]].astype(np.float64)
        mu_lf += blk.sum(axis=0)
        sq_lf += (blk ** 2).sum(axis=0)
    mu_lf /= len(train_idx)
    sd_lf = np.sqrt(np.maximum(sq_lf / len(train_idx) - mu_lf ** 2, 0.0)) + 1e-6
    for s in range(0, n_samples, 5000):
        logfreq_np[s:s + 5000] = ((logfreq_np[s:s + 5000] - mu_lf) / sd_lf).astype(np_dtype)

    p_mu = params_np[train_idx].mean(axis=0, keepdims=True, dtype=np.float64)
    p_sd = np.maximum(params_np[train_idx].std(axis=0, keepdims=True, dtype=np.float64), 1e-8)
    params_norm_np = ((params_np - p_mu) / p_sd).astype(np_dtype)

    norm_path = f"model_weights/{title}_norm_params.npz"
    np.savez(
        norm_path,
        mu_lf=mu_lf,
        sd_lf=sd_lf,
        p_mu=p_mu,
        p_sd=p_sd,
        omega_floor=args.omega_floor,
    )
    print(f"Saved normalization params: {norm_path}")

    # Load exact ground-truth complex test waveforms, true phase, and true log-omega for overlap & error diagnostics
    with h5py.File(filepath, "r") as f:
        wf_ds = f["Waveforms"]
        order = np.argsort(test_idx)
        inv_order = np.argsort(order)
        h_test_sorted = wf_ds[test_idx[order]]
        h_test_np = h_test_sorted[inv_order]
        h_test_true = torch.from_numpy(h_test_np).to(torch.complex128)
        h_test_true = h_test_true * torch.exp(-1j * torch.angle(h_test_true[:, :1]))
        true_phase_test = torch.from_numpy(wave_to_phase(h_test_np)).to(torch.float64)
        true_log_om_test = torch.from_numpy(phase_to_logfreq(true_phase_test.numpy(), omega_floor=args.omega_floor, dtype=np.float64)).to(torch.float64)

    params_t = torch.from_numpy(params_norm_np).to(dtype=dtype)
    ampl_t = torch.from_numpy(ampl_np).to(dtype=dtype)
    logfreq_t = torch.from_numpy(logfreq_np).to(dtype=dtype)

    mu_lf_t = torch.from_numpy(mu_lf).to(device=device, dtype=torch.float64).view(1, -1)
    sd_lf_t = torch.from_numpy(sd_lf).to(device=device, dtype=torch.float64).view(1, -1)

    # Check representation round-trip floor on test set
    with torch.no_grad():
        hp_rt = reconstruct_waveform_torch(
            ampl_t[test_idx[:256]].to(device),
            logfreq_t[test_idx[:256]].to(device),
            mu_lf_t,
            sd_lf_t,
        )
        rt_mm = torch.clamp(1.0 - TasosOverlapTorch(h_test_true[:256].to(device), hp_rt, eps=args.eps), min=0.0)
        print(f"LogFreq representation floor (256 test samples): median={rt_mm.median().item():.3e}, max={rt_mm.max().item():.3e}")

    # 4. Build models from models_mlp
    enc_amp = Encoder(sample_len, args.latent_dim, args.ae_neurons).to(device=device, dtype=dtype)
    dec_amp = Decoder(sample_len, args.latent_dim, args.ae_neurons).to(device=device, dtype=dtype)
    ae_amp = Autoencoder(enc_amp, dec_amp).to(device=device, dtype=dtype)

    enc_lf = Encoder(sample_len, args.latent_dim, args.ae_neurons).to(device=device, dtype=dtype)
    dec_lf = Decoder(sample_len, args.latent_dim, args.ae_neurons).to(device=device, dtype=dtype)
    ae_lf = Autoencoder(enc_lf, dec_lf).to(device=device, dtype=dtype)

    loss_amp_fn = nn.L1Loss()
    loss_lf_fn = AmplitudeWeightedPhaseLoss(mu_lf_t, sd_lf_t, l1_weight=args.lf_l1_weight).to(device=device)
    loss_joint_fn = LogMismatchLoss(mu_lf_t, sd_lf_t, log_eps=args.log_mm_eps, overlap_eps=args.eps).to(device=device)
    meta = {
        "title": title,
        "dataset": filepath,
        "sample_len": sample_len,
        "labels": n_labels,
        "latent_dim": args.latent_dim,
        "ae_neurons": args.ae_neurons,
        "full_neurons": args.full_neurons,
        "f64": True,
        "omega_floor": args.omega_floor,
        "norm_path": norm_path,
    }
    stage_diags: dict[str, dict] = {}
    histories: list[tuple[str, list, list, list]] = []
    pdf_path = f"runs/{title}_plots.pdf"
    pin = torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Stage 1: AE training (Amplitude & Log-Frequency)
    # ------------------------------------------------------------------
    a_tr_loader = DataLoader(TensorDataset(ampl_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    a_va_loader = DataLoader(TensorDataset(ampl_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)
    l_tr_loader = DataLoader(TensorDataset(logfreq_t[train_idx], logfreq_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    l_va_loader = DataLoader(TensorDataset(logfreq_t[val_idx], logfreq_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    exec_ae_amp = compile_model(ae_amp, "AE_Amplitude")
    exec_ae_lf = compile_model(ae_lf, "AE_LogFreq")

    s1_a_tr, s1_a_va, s1_a_lr, s1_a_best = run_branch_stage(
        "Stage 1 [Amplitude AE]", exec_ae_amp, ae_amp, list(ae_amp.parameters()),
        a_tr_loader, a_va_loader, loss_amp_fn, args.ae_epochs, args.lr_ae, args.min_lr_ae, device,
    )
    s1_l_tr, s1_l_va, s1_l_lr, s1_l_best = run_branch_stage(
        "Stage 1 [LogFreq AE]", exec_ae_lf, ae_lf, list(ae_lf.parameters()),
        l_tr_loader, l_va_loader, loss_lf_fn, args.ae_epochs, args.lr_ae, args.min_lr_ae, device,
    )

    if s1_a_best is not None:
        ae_amp.load_state_dict(s1_a_best)
    if s1_l_best is not None:
        ae_lf.load_state_dict(s1_l_best)

    diag_stage1 = evaluate_stage(
        ae_amp, ae_lf, ae_amp.encoder, ae_lf.encoder,
        ampl_t[test_idx], logfreq_t[test_idx], h_test_true, true_phase_test, true_log_om_test,
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    stage_diags["Stage 1: AE"] = diag_stage1
    histories.extend([
        ("Stage 1: Amplitude AE", s1_a_tr, s1_a_va, s1_a_lr),
        ("Stage 1: LogFreq AE", s1_l_tr, s1_l_va, s1_l_lr),
    ])
    print_mismatch_stats("Stage 1: AE", diag_stage1)
    torch.save({**meta, "ae_amp": ae_amp.state_dict(), "ae_lf": ae_lf.state_dict()}, f"model_weights/{title}_stage1_ae_best.pt")
    save_report_pdf(pdf_path, stage_diags, histories, title)

    # ------------------------------------------------------------------
    # Stage 2: Full (decoder frozen)
    # ------------------------------------------------------------------
    full_amp = Full(dec_amp, labels=n_labels, latent_dim=args.latent_dim, neurons_unit=args.full_neurons, trainable=False).to(device=device, dtype=dtype)
    full_lf = Full(dec_lf, labels=n_labels, latent_dim=args.latent_dim, neurons_unit=args.full_neurons, trainable=False).to(device=device, dtype=dtype)

    fa_tr_loader = DataLoader(TensorDataset(params_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    fa_va_loader = DataLoader(TensorDataset(params_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)
    fl_tr_loader = DataLoader(TensorDataset(params_t[train_idx], logfreq_t[train_idx], ampl_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    fl_va_loader = DataLoader(TensorDataset(params_t[val_idx], logfreq_t[val_idx], ampl_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    exec_s2_amp = compile_model(full_amp, "Stage2_Full_Amplitude")
    exec_s2_lf = compile_model(full_lf, "Stage2_Full_LogFreq")

    s2_a_tr, s2_a_va, s2_a_lr, s2_a_best = run_branch_stage(
        "Stage 2 [Full Amplitude | Frozen Dec]", exec_s2_amp, full_amp, list(full_amp.mapper.parameters()),
        fa_tr_loader, fa_va_loader, loss_amp_fn, args.mapper_epochs, args.lr_mapper, args.min_lr_mapper, device,
    )
    s2_l_tr, s2_l_va, s2_l_lr, s2_l_best = run_branch_stage(
        "Stage 2 [Full LogFreq | Frozen Dec]", exec_s2_lf, full_lf, list(full_lf.mapper.parameters()),
        fl_tr_loader, fl_va_loader, loss_lf_fn, args.mapper_epochs, args.lr_mapper, args.min_lr_mapper, device,
    )

    if s2_a_best is not None:
        full_amp.load_state_dict(s2_a_best)
    if s2_l_best is not None:
        full_lf.load_state_dict(s2_l_best)

    diag_stage2 = evaluate_stage(
        full_amp, full_lf, full_amp.mapper, full_lf.mapper,
        params_t[test_idx], params_t[test_idx], h_test_true, true_phase_test, true_log_om_test,
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    stage_diags["Stage 2: Full (Frozen Dec)"] = diag_stage2
    histories.extend([
        ("Stage 2: Full Amp (Frozen)", s2_a_tr, s2_a_va, s2_a_lr),
        ("Stage 2: Full LogFreq (Frozen)", s2_l_tr, s2_l_va, s2_l_lr),
    ])
    print_mismatch_stats("Stage 2: Full (Frozen Dec)", diag_stage2)
    torch.save({**meta, "full_amp": full_amp.state_dict(), "full_lf": full_lf.state_dict()}, f"model_weights/{title}_stage2_full_frozen_best.pt")
    save_report_pdf(pdf_path, stage_diags, histories, title)

    # ------------------------------------------------------------------
    # Stage 3: Joint Full (decoder trainable, direct LogMismatchLoss)
    # ------------------------------------------------------------------
    for p in full_amp.parameters():
        p.requires_grad_(True)
    for p in full_lf.parameters():
        p.requires_grad_(True)

    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    exec_s3_amp = compile_model(full_amp, "Stage3_Full_Amplitude")
    exec_s3_lf = compile_model(full_lf, "Stage3_Full_LogFreq")

    joint_tr_loader = DataLoader(TensorDataset(params_t[train_idx], ampl_t[train_idx], logfreq_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    joint_va_loader = DataLoader(TensorDataset(params_t[val_idx], ampl_t[val_idx], logfreq_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    s3_tr, s3_va, s3_lr, s3_a_best, s3_l_best = run_joint_stage(
        "Stage 3 [Joint Full | Trainable Dec | LogMismatch]",
        exec_s3_amp, exec_s3_lf, full_amp, full_lf,
        joint_tr_loader, joint_va_loader, loss_joint_fn,
        args.joint_epochs, args.lr_joint, args.min_lr_joint, device,
    )

    if s3_a_best is not None:
        full_amp.load_state_dict(s3_a_best)
    if s3_l_best is not None:
        full_lf.load_state_dict(s3_l_best)

    diag_stage3 = evaluate_stage(
        full_amp, full_lf, full_amp.mapper, full_lf.mapper,
        params_t[test_idx], params_t[test_idx], h_test_true, true_phase_test, true_log_om_test,
        mu_lf_t, sd_lf_t, args.val_batch_size, device, args.eps,
    )
    stage_diags["Stage 3: Full (Trainable Dec)"] = diag_stage3
    histories.append(("Stage 3: Joint Full (LogMismatch)", s3_tr, s3_va, s3_lr))
    print_mismatch_stats("Stage 3: Full (Trainable Dec)", diag_stage3)
    torch.save({**meta, "full_amp": full_amp.state_dict(), "full_lf": full_lf.state_dict()}, f"model_weights/{title}_stage3_full_trainable_best.pt")

    # ------------------------------------------------------------------
    # Save final vector PDF report, mismatches, and training history
    # ------------------------------------------------------------------
    npz_path = f"runs/{title}_mismatches.npz"
    save_report_pdf(pdf_path, stage_diags, histories, title)
    np.savez_compressed(
        npz_path,
        stage1_ae=diag_stage1["mm"],
        stage2_full_frozen=diag_stage2["mm"],
        stage3_full_trainable=diag_stage3["mm"],
    )

    rows = []
    for stage_tag, (tr, va, lr) in [
        ("stage1_amp_ae", (s1_a_tr, s1_a_va, s1_a_lr)),
        ("stage1_lf_ae", (s1_l_tr, s1_l_va, s1_l_lr)),
        ("stage2_amp_full_frozen", (s2_a_tr, s2_a_va, s2_a_lr)),
        ("stage2_lf_full_frozen", (s2_l_tr, s2_l_va, s2_l_lr)),
        ("stage3_joint_full_logmismatch", (s3_tr, s3_va, s3_lr)),
    ]:
        for ep_i, (t_v, v_v, l_v) in enumerate(zip(tr, va, lr), 1):
            rows.append({"stage": stage_tag, "epoch": ep_i, "train_loss": t_v, "val_loss": v_v, "lr": l_v})
    pd.DataFrame(rows).to_csv(f"runs/{title}_metrics.csv", index=False)

    print("\nSaved artifacts:")
    print(f"  - model_weights/{title}_norm_params.npz")
    print(f"  - model_weights/{title}_stage1_ae_best.pt")
    print(f"  - model_weights/{title}_stage2_full_frozen_best.pt")
    print(f"  - model_weights/{title}_stage3_full_trainable_best.pt")
    print(f"  - {pdf_path}")
    print(f"  - {npz_path}")
    print(f"  - runs/{title}_metrics.csv")


if __name__ == "__main__":
    main()
