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

import argparse
import os
import sys
import time
from datetime import datetime

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np
import pandas as pd
import pytorch_optimizer as opt_add
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split

from models_mlp import Decoder, Encoder, Full
from utils import TasosOverlapTorch


# ---------------------------------------------------------------------------
# Precision & Hardware Guards (always float64, disable precision shortcuts)
# ---------------------------------------------------------------------------
PRECONDITION_FREQ = 15
WEIGHT_DECAY = 0.0
SOAP_EPS = 1e-14


def configure_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_default_dtype(torch.float64)
    print("Precision: float64 (TF32 & reduced-precision reductions disabled).")


# ---------------------------------------------------------------------------
# Autoencoder Wrapper (models_mlp keeps Encoder, Decoder, Full separate)
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Phase <-> Standardized Log-Frequency Helpers
# ---------------------------------------------------------------------------
def wave_to_phase(w: np.ndarray) -> np.ndarray:
    ph = -np.unwrap(np.angle(w), axis=-1)
    return ph - ph[:, :1]


def phase_to_logfreq(ph: np.ndarray, omega_floor: float = 1e-6, dtype=np.float64) -> np.ndarray:
    om = np.diff(ph, axis=-1)
    om = np.concatenate([om, om[:, -1:]], axis=-1)
    om = np.maximum(om, omega_floor)
    return np.log(om).astype(dtype)


def _pred_waveform_parts(
    ampl: torch.Tensor,
    logfreq_norm: torch.Tensor,
    mu_lf: torch.Tensor,
    sd_lf: torch.Tensor,
):
    """Return (complex_wf, amp_pos, phase, log_om) in double precision."""
    log_om = logfreq_norm.to(torch.float64) * sd_lf + mu_lf
    om = torch.exp(torch.clamp(log_om, max=20.0))
    zeros = torch.zeros((om.shape[0], 1), dtype=torch.float64, device=om.device)
    phase = torch.cat([zeros, torch.cumsum(om[:, :-1], dim=-1)], dim=-1)
    amp_pos = torch.clamp(ampl.to(torch.float64), min=0.0)
    return torch.polar(amp_pos, -phase), amp_pos, phase, log_om


def reconstruct_waveform_torch(
    ampl: torch.Tensor,
    logfreq_norm: torch.Tensor,
    mu_lf: torch.Tensor,
    sd_lf: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct complex waveform in complex128 from amplitude and normalized log-frequency."""
    wf, _, _, _ = _pred_waveform_parts(ampl, logfreq_norm, mu_lf, sd_lf)
    return wf


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="3-stage Amplitude + Log-Frequency MLP training (f64, L1, SOAP, CosineAnnealingLR, torch.compile)"
    )
    parser.add_argument("--title", type=str, default=None, help="Run prefix for checkpoints and plots")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/NRSur7dq4v2_dataset.hdf",
        help="HDF5 dataset path (default: datasets/NRSur7dq4v2_dataset.hdf)",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on loaded samples")

    # Architecture defaults from models_mlp.py
    parser.add_argument("--latent_dim", type=int, default=24, help="Latent dimension (default: 24)")
    parser.add_argument("--ae_neurons", type=int, default=128, help="Encoder/Decoder neurons_unit (default: 128)")
    parser.add_argument("--full_neurons", type=int, default=64, help="Full mapper neurons_unit (default: 64)")

    # Training schedule (no early stopping, 3000 epochs per stage)
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size (default: 256)")
    parser.add_argument("--val_batch_size", type=int, default=2048, help="Validation/test batch size (default: 2048)")
    parser.add_argument("--ae_epochs", type=int, default=3000, help="Stage 1 AE epochs (default: 3000)")
    parser.add_argument("--mapper_epochs", type=int, default=3000, help="Stage 2 Full frozen-decoder epochs (default: 3000)")
    parser.add_argument("--joint_epochs", type=int, default=3000, help="Stage 3 Full trainable-decoder epochs (default: 3000)")

    # Learning rates
    parser.add_argument("--lr_ae", type=float, default=1e-3, help="Stage 1 initial LR (default: 1e-3)")
    parser.add_argument("--min_lr_ae", type=float, default=1e-7, help="Stage 1 cosine min LR (default: 1e-7)")
    parser.add_argument("--lr_mapper", type=float, default=1e-3, help="Stage 2 initial LR (default: 1e-3)")
    parser.add_argument("--min_lr_mapper", type=float, default=1e-7, help="Stage 2 cosine min LR (default: 1e-7)")
    parser.add_argument("--lr_joint", type=float, default=5e-5, help="Stage 3 initial LR (default: 5e-5)")
    parser.add_argument("--min_lr_joint", type=float, default=5e-8, help="Stage 3 cosine min LR (default: 5e-8)")

    parser.add_argument("--omega_floor", type=float, default=1e-6, help="Floor on instantaneous omega before log (default: 1e-6)")
    parser.add_argument("--eps", type=float, default=1e-30, help="eps for TasosOverlapTorch evaluation (default: 1e-30)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers: Optimizer, Compile, Train Stage, Evaluation, Plotting
# ---------------------------------------------------------------------------
def compile_model(model: nn.Module, label: str) -> nn.Module:
    if sys.version_info >= (3, 13):
        print(f"Notice: torch.compile not supported on Python 3.13+; running {label} eager.")
        return model
    try:
        compiled = torch.compile(model)
        print(f"Compiled {label} with torch.compile.")
        return compiled
    except Exception as e:
        print(f"Notice: torch.compile failed for {label} ({e}); running eager.")
        return model


def make_soap_and_cosine(params, lr: float, min_lr: float, total_steps: int):
    optimizer = opt_add.SOAP(
        params,
        lr=lr,
        betas=(0.95, 0.95),
        shampoo_beta=0.99,
        weight_decay=WEIGHT_DECAY,
        precondition_frequency=PRECONDITION_FREQ,
        merge_dims=False,
        eps=SOAP_EPS,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=min_lr)
    return optimizer, scheduler


def run_branch_stage(
    stage_label: str,
    exec_model: nn.Module,
    raw_model: nn.Module,
    trainable_params,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    epochs: int,
    lr: float,
    min_lr: float,
    device: torch.device,
):
    if epochs <= 0:
        return [], [], [], None

    total_steps = epochs * len(train_loader)
    optimizer, scheduler = make_soap_and_cosine(trainable_params, lr, min_lr, total_steps)

    best_val = float("inf")
    best_state = None
    tr_hist, va_hist, lr_hist = [], [], []
    log_every = max(1, epochs // 20)
    n_params = sum(p.numel() for p in trainable_params)

    print(f"\n--- {stage_label} | epochs={epochs} | lr={lr:.1e} -> {min_lr:.1e} (cosine) | trainable_params={n_params:,} ---")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        exec_model.train(True)
        tr_sum = 0.0
        for x_b, y_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = exec_model(x_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_sum += float(loss.item()) * len(x_b)
        tr_loss = tr_sum / len(train_loader.dataset)

        exec_model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                va_sum += float(loss_fn(exec_model(x_b), y_b).item()) * len(x_b)
        va_loss = va_sum / len(val_loader.dataset)
        cur_lr = optimizer.param_groups[0]["lr"]

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)
        lr_hist.append(cur_lr)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:4d}/{epochs} | train={tr_loss:.4e} | val={va_loss:.4e} | "
                f"best={best_val:.4e} | lr={cur_lr:.2e}"
            )

    print(f"--- {stage_label} finished in {time.time() - t0:.1f}s (best val={best_val:.4e}) ---")
    return tr_hist, va_hist, lr_hist, best_state


def evaluate_stage(
    model_amp: nn.Module,
    model_lf: nn.Module,
    latent_amp_fn: nn.Module,
    latent_lf_fn: nn.Module,
    in_amp: torch.Tensor,
    in_lf: torch.Tensor,
    h_true: torch.Tensor,
    true_phase: torch.Tensor,
    true_log_om: torch.Tensor,
    mu_lf: torch.Tensor,
    sd_lf: torch.Tensor,
    batch_size: int,
    device: torch.device,
    eps: float,
) -> dict:
    """Evaluate test-set mismatches, time-resolved errors, and latent representations."""
    model_amp.eval()
    model_lf.eval()
    latent_amp_fn.eval()
    latent_lf_fn.eval()

    mismatches = []
    amp_errs, lf_errs, ph_errs, wf_errs = [], [], [], []
    z_amps, z_lfs = [], []

    with torch.no_grad():
        for s in range(0, len(h_true), batch_size):
            xa = in_amp[s:s + batch_size].to(device)
            xl = in_lf[s:s + batch_size].to(device)
            ht = h_true[s:s + batch_size].to(device)
            pht = true_phase[s:s + batch_size].to(device)
            lft = true_log_om[s:s + batch_size].to(device)

            pred_amp = model_amp(xa)
            pred_lf = model_lf(xl)
            z_amps.append(latent_amp_fn(xa).cpu().numpy())
            z_lfs.append(latent_lf_fn(xl).cpu().numpy())

            hp, amp_p, ph_p, lf_p = _pred_waveform_parts(pred_amp, pred_lf, mu_lf, sd_lf)
            ov = TasosOverlapTorch(ht, hp, eps=eps)
            mismatches.append(torch.clamp(1.0 - ov, min=0.0).cpu().numpy())

            amp_errs.append(torch.abs(amp_p - torch.abs(ht)).cpu().numpy().astype(np.float32))
            lf_errs.append(torch.abs(lf_p - lft).cpu().numpy().astype(np.float32))
            ph_errs.append(torch.abs(ph_p - pht).cpu().numpy().astype(np.float32))
            wf_errs.append(torch.abs(hp - ht).cpu().numpy().astype(np.float32))

    amp_err_all = np.concatenate(amp_errs, axis=0)
    lf_err_all = np.concatenate(lf_errs, axis=0)
    ph_err_all = np.concatenate(ph_errs, axis=0)
    wf_err_all = np.concatenate(wf_errs, axis=0)

    L = wf_err_all.shape[1]
    t1, t2 = L // 3, (2 * L) // 3
    region_wf_med = [
        float(np.median(wf_err_all[:, :t1])),
        float(np.median(wf_err_all[:, t1:t2])),
        float(np.median(wf_err_all[:, t2:])),
    ]
    region_wf_p95 = [
        float(np.percentile(wf_err_all[:, :t1], 95)),
        float(np.percentile(wf_err_all[:, t1:t2], 95)),
        float(np.percentile(wf_err_all[:, t2:], 95)),
    ]

    return {
        "mm": np.concatenate(mismatches).astype(np.float64),
        "amp_med": np.median(amp_err_all, axis=0),
        "amp_p95": np.percentile(amp_err_all, 95, axis=0),
        "lf_med": np.median(lf_err_all, axis=0),
        "lf_p95": np.percentile(lf_err_all, 95, axis=0),
        "ph_med": np.median(ph_err_all, axis=0),
        "ph_p95": np.percentile(ph_err_all, 95, axis=0),
        "wf_med": np.median(wf_err_all, axis=0),
        "wf_p95": np.percentile(wf_err_all, 95, axis=0),
        "region_wf_med": region_wf_med,
        "region_wf_p95": region_wf_p95,
        "z_amp": np.concatenate(z_amps, axis=0).astype(np.float64),
        "z_lf": np.concatenate(z_lfs, axis=0).astype(np.float64),
    }


def print_mismatch_stats(label: str, diag: dict):
    mm = diag["mm"]
    r_med = diag["region_wf_med"]
    print(
        f"  [Eval] {label:<32} | med={np.median(mm):.3e} | mean={np.mean(mm):.3e} | "
        f"p95={np.percentile(mm, 95):.3e} | p99={np.percentile(mm, 99):.3e} | max={np.max(mm):.3e} | "
        f"<1e-3={np.mean(mm < 1e-3) * 100:.1f}% | <1e-4={np.mean(mm < 1e-4) * 100:.1f}%"
    )
    print(
        f"         Region median |dh|: Start={r_med[0]:.3e} | Mid={r_med[1]:.3e} | End={r_med[2]:.3e}"
    )


def build_single_stage_mismatch_fig(mm: np.ndarray, stage_title: str, color: str = "#2b5c8f"):
    vals = np.maximum(mm, 1e-16)
    min_exp = np.floor(np.log10(np.min(vals))) - 0.3
    max_exp = max(np.ceil(np.log10(np.max(vals))) + 0.3, min_exp + 2.0)
    bins = np.logspace(min_exp, max_exp, 65)

    med, mean_v = float(np.median(mm)), float(np.mean(mm))
    p95, p99, max_v = float(np.percentile(mm, 95)), float(np.percentile(mm, 99)), float(np.max(mm))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(vals, bins=bins, color=color, edgecolor="#172b4d", alpha=0.82)
    ax.axvline(med, color="#d62728", ls="-", lw=2.0, label=f"Median: {med:.2e}")
    ax.axvline(p95, color="#ff7f0e", ls="--", lw=1.8, label=f"p95: {p95:.2e}")
    ax.axvline(p99, color="#9467bd", ls="-.", lw=1.8, label=f"p99: {p99:.2e}")
    ax.axvline(max_v, color="black", ls=":", lw=1.5, label=f"Max: {max_v:.2e}")

    ax.set_xscale("log")
    ax.set_xlabel("Test-Set Waveform Mismatch (1 - Overlap)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title(stage_title, fontsize=11.5, fontweight="bold")

    stats_text = (
        f"N:      {len(mm)}\n"
        f"Median: {med:.2e}\n"
        f"Mean:   {mean_v:.2e}\n"
        f"p95:    {p95:.2e}\n"
        f"p99:    {p99:.2e}\n"
        f"Max:    {max_v:.2e}\n"
        f"<1e-3:  {np.mean(mm < 1e-3) * 100:.1f}%"
    )
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", edgecolor=color, facecolor="#f4f7fb", alpha=0.92),
    )
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def build_mismatch_comparison_fig(mm_dict: dict[str, np.ndarray], title: str):
    all_vals = np.concatenate([np.maximum(v, 1e-16) for v in mm_dict.values()])
    min_exp = np.floor(np.log10(np.min(all_vals))) - 0.3
    max_exp = max(np.ceil(np.log10(np.max(all_vals))) + 0.3, min_exp + 2.0)
    bins = np.logspace(min_exp, max_exp, 65)

    colors = {
        "Stage 1: AE": "#1f77b4",
        "Stage 2: Full (Frozen Dec)": "#ff7f0e",
        "Stage 3: Full (Trainable Dec)": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for label, mm in mm_dict.items():
        med = float(np.median(mm))
        p95 = float(np.percentile(mm, 95))
        c = colors.get(label, "#2ca02c")
        ax.hist(
            np.maximum(mm, 1e-16),
            bins=bins,
            histtype="step",
            lw=2.0,
            color=c,
            label=f"{label} (med={med:.2e}, p95={p95:.2e})",
        )
        ax.axvline(med, color=c, ls=":", lw=1.3)

    ax.set_xscale("log")
    ax.set_xlabel("Test-Set Waveform Mismatch (1 - Overlap)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(f"Test-Set Mismatch: AE vs Full — {title}", fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def build_training_curves_fig(histories: list[tuple[str, list, list, list]], title: str):
    active = [h for h in histories if len(h[1]) > 0]
    if not active:
        return None
    n = len(active)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (name, tr, va, lrs) in zip(axes_flat, active):
        ep = np.arange(1, len(tr) + 1)
        ax.semilogy(ep, tr, label="Train", color="#1f77b4", lw=1.4)
        ax.semilogy(ep, va, label="Val", color="#d62728", lw=1.4)
        ax_lr = ax.twinx()
        ax_lr.semilogy(ep, lrs, label="LR", color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax_lr.set_ylabel("LR")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        lines = ax.get_lines() + ax_lr.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    for ax in axes_flat[len(active):]:
        ax.axis("off")

    fig.suptitle(f"MLP Training Curves — {title}", fontweight="bold")
    fig.tight_layout()
    return fig


def _latent_metrics(z: np.ndarray) -> dict:
    """Compute std, singular values, effective rank, and off-diagonal correlations."""
    n_bad = int((~np.isfinite(z)).sum())
    z_clean = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    stds = z_clean.std(axis=0)
    dead = int((stds < 1e-3).sum())
    max_abs = float(np.max(np.abs(z_clean)))

    zc = z_clean - z_clean.mean(axis=0, keepdims=True)
    s = np.linalg.svd(zc, compute_uv=False)
    s_norm = s / max(float(s[0]), 1e-12)
    p = (s ** 2) / max(float((s ** 2).sum()), 1e-24)
    p_pos = p[p > 1e-15]
    eff_rank = float(np.exp(-np.sum(p_pos * np.log(p_pos)))) if len(p_pos) else 0.0

    active = stds > 1e-12
    if active.sum() >= 2:
        corr = np.abs(np.corrcoef(z_clean[:, active], rowvar=False))
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_off_corr = float(np.mean(off))
        max_off_corr = float(np.max(off))
    else:
        mean_off_corr, max_off_corr = 0.0, 0.0

    return {
        "stds": stds,
        "dead": dead,
        "n_bad": n_bad,
        "max_abs": max_abs,
        "s_norm": s_norm,
        "eff_rank": eff_rank,
        "mean_off_corr": mean_off_corr,
        "max_off_corr": max_off_corr,
    }


def build_error_profile_fig(stage_diags: dict[str, dict], title: str):
    """Diagnose where error lives over waveform time: Start vs Middle vs End."""
    stage_colors = {
        "Stage 1: AE": "#1f77b4",
        "Stage 2: Full (Frozen Dec)": "#ff7f0e",
        "Stage 3: Full (Trainable Dec)": "#d62728",
    }
    first_diag = next(iter(stage_diags.values()))
    L = len(first_diag["wf_med"])
    t = np.arange(L)
    t1, t2 = L // 3, (2 * L) // 3

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    panels = [
        (axes[0, 0], "amp_med", "amp_p95", "Amplitude Error |A_pred - A_true|"),
        (axes[0, 1], "lf_med", "lf_p95", "Log-Frequency Error |log(w_pred) - log(w_true)|"),
        (axes[1, 0], "ph_med", "ph_p95", "Cumulative Phase Error |phi_pred - phi_true| (rad)"),
    ]
    for ax, k_med, k_p95, ylabel in panels:
        for label, d in stage_diags.items():
            c = stage_colors.get(label, "#2ca02c")
            ax.semilogy(t, np.maximum(d[k_med], 1e-16), color=c, lw=1.6, label=f"{label} (med)")
            ax.semilogy(t, np.maximum(d[k_p95], 1e-16), color=c, lw=0.9, ls="--", alpha=0.65)
        ax.axvline(t1, color="#555555", ls=":", lw=1.1)
        ax.axvline(t2, color="#555555", ls=":", lw=1.1)
        ax.axvspan(0, t1, color="#eef4fb", alpha=0.35, zorder=0)
        ax.axvspan(t2, L, color="#fdf0ef", alpha=0.35, zorder=0)
        ax.set_xlabel("Time Sample Index (Start | Middle | End)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left")

    ax_r = axes[1, 1]
    regions = ["Start\n[0, L/3)", "Middle\n[L/3, 2L/3)", "End\n[2L/3, L)"]
    x_pos = np.arange(3)
    n_stages = len(stage_diags)
    width = 0.72 / max(n_stages, 1)
    for idx, (label, d) in enumerate(stage_diags.items()):
        c = stage_colors.get(label, "#2ca02c")
        offset = (idx - (n_stages - 1) / 2.0) * width
        vals = np.maximum(d["region_wf_med"], 1e-16)
        bars = ax_r.bar(x_pos + offset, vals, width=width * 0.92, color=c, alpha=0.88, label=label)
        for b, v in zip(bars, vals):
            ax_r.text(b.get_x() + b.get_width() / 2.0, v * 1.12, f"{v:.1e}", ha="center", va="bottom", fontsize=7, rotation=35)

    ax_r.set_yscale("log")
    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(regions)
    ax_r.set_ylabel("Median |h_pred - h_true|")
    ax_r.set_title("Complex Waveform Error by Region (Solid=Med, Dashed=p95 in time plots)", fontweight="bold", fontsize=9.5)
    ax_r.grid(True, which="both", ls="--", alpha=0.3, axis="y")
    ax_r.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(f"Where Is the Error? Time-Resolved Diagnostics — {title}", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    return fig


def build_latent_sanity_fig(stage_diags: dict[str, dict], title: str):
    """Sanity checks on latent space: activity, collapse/rank, correlation, and mapper alignment."""
    stage_colors = {
        "Stage 1: AE": "#1f77b4",
        "Stage 2: Full (Frozen Dec)": "#ff7f0e",
        "Stage 3: Full (Trainable Dec)": "#d62728",
    }
    first_diag = next(iter(stage_diags.values()))
    lat_dim = first_diag["z_amp"].shape[1]
    dims = np.arange(lat_dim)

    metrics = {
        label: {"amp": _latent_metrics(d["z_amp"]), "lf": _latent_metrics(d["z_lf"])}
        for label, d in stage_diags.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    # Top row: per-dimension std for Amplitude and LogFreq
    for ax, branch_key, branch_name in [
        (axes[0, 0], "amp", "Amplitude Latent Std per Dimension"),
        (axes[0, 1], "lf", "Log-Frequency Latent Std per Dimension"),
    ]:
        for label, m in metrics.items():
            c = stage_colors.get(label, "#2ca02c")
            src = "Encoder" if "Stage 1" in label else "Mapper"
            ax.plot(dims, m[branch_key]["stds"], marker="o", ms=3.5, lw=1.4, color=c, label=f"{label} ({src}, dead={m[branch_key]['dead']})")
        ax.axhline(1e-3, color="red", ls="--", lw=1.0, alpha=0.7, label="Dead threshold (1e-3)")
        ax.set_yscale("log")
        ax.set_xlabel("Latent Dimension Index")
        ax.set_ylabel("Std across Test Set")
        ax.set_title(branch_name, fontweight="bold", fontsize=10.5)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    # Bottom-left: singular value spectrum (collapse / effective rank check)
    ax_svd = axes[1, 0]
    sv_idx = np.arange(1, lat_dim + 1)
    for label, m in metrics.items():
        c = stage_colors.get(label, "#2ca02c")
        ax_svd.semilogy(sv_idx, np.maximum(m["amp"]["s_norm"], 1e-12), color=c, lw=1.6, ls="-", label=f"{label} Amp (rank={m['amp']['eff_rank']:.1f})")
        ax_svd.semilogy(sv_idx, np.maximum(m["lf"]["s_norm"], 1e-12), color=c, lw=1.4, ls="--", label=f"{label} LF (rank={m['lf']['eff_rank']:.1f})")
    ax_svd.set_xlabel("Singular Value Index")
    ax_svd.set_ylabel("Normalized Singular Value (s_i / s_0)")
    ax_svd.set_title("Latent SVD Spectrum & Effective Rank", fontweight="bold", fontsize=10.5)
    ax_svd.grid(True, which="both", ls="--", alpha=0.3)
    ax_svd.legend(fontsize=7, ncol=2, loc="lower left")

    # Bottom-right: Mapper vs Encoder R^2 per dim (if Stage 2/3 present) + health summary
    ax_al = axes[1, 1]
    ae_diag = stage_diags.get("Stage 1: AE")
    has_mapper = False
    if ae_diag is not None:
        z_enc_a, z_enc_l = ae_diag["z_amp"], ae_diag["z_lf"]
        var_a = np.maximum(z_enc_a.var(axis=0), 1e-24)
        var_l = np.maximum(z_enc_l.var(axis=0), 1e-24)
        for label, d in stage_diags.items():
            if "Stage 1" in label:
                continue
            has_mapper = True
            c = stage_colors.get(label, "#2ca02c")
            r2_a = 1.0 - np.mean((d["z_amp"] - z_enc_a) ** 2, axis=0) / var_a
            r2_l = 1.0 - np.mean((d["z_lf"] - z_enc_l) ** 2, axis=0) / var_l
            ax_al.plot(dims, np.clip(r2_a, -1.0, 1.0), color=c, lw=1.5, ls="-", marker="s", ms=3, label=f"{label} Amp (med R2={np.median(r2_a):.3f})")
            ax_al.plot(dims, np.clip(r2_l, -1.0, 1.0), color=c, lw=1.3, ls="--", marker="^", ms=3, label=f"{label} LF (med R2={np.median(r2_l):.3f})")

    if has_mapper:
        ax_al.axhline(0.0, color="black", ls=":", lw=1.0)
        ax_al.set_ylim(-0.2, 1.05)
        ax_al.set_xlabel("Latent Dimension Index")
        ax_al.set_ylabel("R^2 (Mapper vs Stage 1 Encoder)")
        ax_al.set_title("Mapper-to-Encoder Latent Alignment (R^2)", fontweight="bold", fontsize=10.5)
        ax_al.grid(True, ls="--", alpha=0.3)
        ax_al.legend(fontsize=7.5, loc="lower left")
    else:
        ax_al.axis("off")

    lines = ["Latent Health Summary:"]
    for label, m in metrics.items():
        ma, ml = m["amp"], m["lf"]
        lines.append(
            f"{label}: Amp[dead={ma['dead']}, bad={ma['n_bad']}, |r|={ma['mean_off_corr']:.2f}, max|z|={ma['max_abs']:.1f}] | "
            f"LF[dead={ml['dead']}, bad={ml['n_bad']}, |r|={ml['mean_off_corr']:.2f}, max|z|={ml['max_abs']:.1f}]"
        )
    fig.suptitle(f"Latent Space Sanity Diagnostics — {title}", fontweight="bold", fontsize=12.5)
    fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", fontsize=7.5, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.35", edgecolor="#cccccc", facecolor="#f8f9fa"))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    return fig


def save_report_pdf(
    pdf_path: str,
    stage_diags: dict[str, dict],
    histories: list[tuple[str, list, list, list]],
    title: str,
):
    """Write all available mismatch, diagnostic, latent-sanity, and training-curve figures into one vector PDF."""
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    stage_colors = {
        "Stage 1: AE": "#1f77b4",
        "Stage 2: Full (Frozen Dec)": "#ff7f0e",
        "Stage 3: Full (Trainable Dec)": "#d62728",
    }
    mm_dict = {k: v["mm"] for k, v in stage_diags.items()}
    with PdfPages(pdf_path) as pdf:
        if len(mm_dict) >= 2:
            fig = build_mismatch_comparison_fig(mm_dict, title)
            pdf.savefig(fig)
            plt.close(fig)

        for label, mm in mm_dict.items():
            fig = build_single_stage_mismatch_fig(
                mm, f"{label} Mismatch ({title})", color=stage_colors.get(label, "#2b5c8f")
            )
            pdf.savefig(fig)
            plt.close(fig)

        if stage_diags:
            err_fig = build_error_profile_fig(stage_diags, title)
            pdf.savefig(err_fig)
            plt.close(err_fig)

            lat_fig = build_latent_sanity_fig(stage_diags, title)
            pdf.savefig(lat_fig)
            plt.close(lat_fig)

        curves_fig = build_training_curves_fig(histories, title)
        if curves_fig is not None:
            pdf.savefig(curves_fig)
            plt.close(curves_fig)


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

    loss_fn = nn.L1Loss()
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
    l_tr_loader = DataLoader(TensorDataset(logfreq_t[train_idx], logfreq_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    l_va_loader = DataLoader(TensorDataset(logfreq_t[val_idx], logfreq_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    exec_ae_amp = compile_model(ae_amp, "AE_Amplitude")
    exec_ae_lf = compile_model(ae_lf, "AE_LogFreq")

    s1_a_tr, s1_a_va, s1_a_lr, s1_a_best = run_branch_stage(
        "Stage 1 [Amplitude AE]", exec_ae_amp, ae_amp, list(ae_amp.parameters()),
        a_tr_loader, a_va_loader, loss_fn, args.ae_epochs, args.lr_ae, args.min_lr_ae, device,
    )
    s1_l_tr, s1_l_va, s1_l_lr, s1_l_best = run_branch_stage(
        "Stage 1 [LogFreq AE]", exec_ae_lf, ae_lf, list(ae_lf.parameters()),
        l_tr_loader, l_va_loader, loss_fn, args.ae_epochs, args.lr_ae, args.min_lr_ae, device,
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
    fl_tr_loader = DataLoader(TensorDataset(params_t[train_idx], logfreq_t[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    fl_va_loader = DataLoader(TensorDataset(params_t[val_idx], logfreq_t[val_idx]), batch_size=args.val_batch_size, shuffle=False, pin_memory=pin)

    exec_s2_amp = compile_model(full_amp, "Stage2_Full_Amplitude")
    exec_s2_lf = compile_model(full_lf, "Stage2_Full_LogFreq")

    s2_a_tr, s2_a_va, s2_a_lr, s2_a_best = run_branch_stage(
        "Stage 2 [Full Amplitude | Frozen Dec]", exec_s2_amp, full_amp, list(full_amp.mapper.parameters()),
        fa_tr_loader, fa_va_loader, loss_fn, args.mapper_epochs, args.lr_mapper, args.min_lr_mapper, device,
    )
    s2_l_tr, s2_l_va, s2_l_lr, s2_l_best = run_branch_stage(
        "Stage 2 [Full LogFreq | Frozen Dec]", exec_s2_lf, full_lf, list(full_lf.mapper.parameters()),
        fl_tr_loader, fl_va_loader, loss_fn, args.mapper_epochs, args.lr_mapper, args.min_lr_mapper, device,
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
    # Stage 3: Full (decoder trainable)
    # ------------------------------------------------------------------
    for p in full_amp.parameters():
        p.requires_grad_(True)
    for p in full_lf.parameters():
        p.requires_grad_(True)

    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    exec_s3_amp = compile_model(full_amp, "Stage3_Full_Amplitude")
    exec_s3_lf = compile_model(full_lf, "Stage3_Full_LogFreq")

    s3_a_tr, s3_a_va, s3_a_lr, s3_a_best = run_branch_stage(
        "Stage 3 [Full Amplitude | Trainable Dec]", exec_s3_amp, full_amp, list(full_amp.parameters()),
        fa_tr_loader, fa_va_loader, loss_fn, args.joint_epochs, args.lr_joint, args.min_lr_joint, device,
    )
    s3_l_tr, s3_l_va, s3_l_lr, s3_l_best = run_branch_stage(
        "Stage 3 [Full LogFreq | Trainable Dec]", exec_s3_lf, full_lf, list(full_lf.parameters()),
        fl_tr_loader, fl_va_loader, loss_fn, args.joint_epochs, args.lr_joint, args.min_lr_joint, device,
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
    histories.extend([
        ("Stage 3: Full Amp (Trainable)", s3_a_tr, s3_a_va, s3_a_lr),
        ("Stage 3: Full LogFreq (Trainable)", s3_l_tr, s3_l_va, s3_l_lr),
    ])
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
        ("stage3_amp_full_trainable", (s3_a_tr, s3_a_va, s3_a_lr)),
        ("stage3_lf_full_trainable", (s3_l_tr, s3_l_va, s3_l_lr)),
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
