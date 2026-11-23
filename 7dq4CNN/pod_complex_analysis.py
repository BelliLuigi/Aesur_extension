#!/usr/bin/env python3
"""
Proper Orthogonal Decomposition (POD) spectrum and mismatch analysis
for the complex128 inertial (2, 2) waveforms in NRSur7dq4v2_dataset_small.hdf.
"""
import argparse
import os
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import TasosOverlapTorch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute complex128 POD spectrum and test-set mismatch curve for NRSur7dq4v2 small dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/NRSur7dq4v2_dataset_small.hdf",
        help="Path to HDF5 dataset (default: datasets/NRSur7dq4v2_dataset_small.hdf)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Waveforms",
        help="Complex128 HDF5 dataset key (default: Waveforms)",
    )
    parser.add_argument(
        "--centered",
        action="store_true",
        help="Subtract training mean before POD (default: uncentered)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-30,
        help="Regularization eps passed to TasosOverlapTorch (default: 1e-30 for complex128 precision)",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="runs/pod_complex128_small_analysis.png",
        help="Output figure path (default: runs/pod_complex128_small_analysis.png)",
    )
    parser.add_argument(
        "--dist_rank",
        type=int,
        default=256,
        help="POD rank k to plot detailed mismatch histogram (default: 256)",
    )
    parser.add_argument(
        "--dist_plot",
        type=str,
        default=None,
        help="Output path for single-rank mismatch distribution plot (default: runs/pod_k<dist_rank>_mismatch_distribution.png)",
    )
    return parser.parse_args()


def get_split_indices(n_samples: int, seed: int = 42):
    idx_all = np.arange(n_samples)
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)
    return (
        np.sort(np.array(train_idx.indices)),
        np.sort(np.array(val_idx.indices)),
        np.sort(np.array(test_idx.indices)),
    )


def compute_pod(X_train: np.ndarray, centered: bool = False):
    """Compute complex POD modes via SVD on (N_train, L) complex128 matrix."""
    if centered:
        mean_wf = X_train.mean(axis=0, keepdims=True)
        X_proc = X_train - mean_wf
    else:
        mean_wf = np.zeros((1, X_train.shape[1]), dtype=X_train.dtype)
        X_proc = X_train

    # Thin QR + SVD on (L, L) R-factor gives exact complex128 singular values & Vh without forming U
    _, R = np.linalg.qr(X_proc, mode="reduced")
    _, s, Vh = np.linalg.svd(R, full_matrices=False)
    energy = s.astype(np.float64) ** 2
    energy_frac = energy / np.sum(energy)
    cum_energy = np.cumsum(energy_frac)
    residual_energy = np.maximum(1.0 - cum_energy, 1e-18)
    return s, energy_frac, cum_energy, residual_energy, Vh, mean_wf


def evaluate_mismatches(X_test: np.ndarray, Vh: np.ndarray, mean_wf: np.ndarray, ranks: list[int], eps: float):
    """Project test waveforms onto top-k complex POD modes and compute physical mismatch."""
    X_test_c = X_test - mean_wf
    # Project once onto all modes up to max(ranks) for speed
    max_k = max(ranks)
    basis_max = Vh[:max_k]  # (max_k, L), orthonormal rows: Vh @ Vh^H = I
    coeffs_all = X_test_c @ basis_max.conj().T  # (N_test, max_k)

    h_true = torch.from_numpy(X_test)
    results = {}
    for k in ranks:
        X_rec = coeffs_all[:, :k] @ basis_max[:k] + mean_wf
        h_rec = torch.from_numpy(X_rec)
        overlaps = TasosOverlapTorch(h_true, h_rec, eps=eps)
        mm = torch.clamp(1.0 - overlaps, min=0.0).numpy()
        results[k] = mm
    return results


def make_plot(energy_frac, residual_energy, mismatch_by_k, output_plot: str, centered: bool):
    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    ranks = sorted(mismatch_by_k.keys())
    k_axis = np.arange(1, len(energy_frac) + 1)

    meds = [np.median(mismatch_by_k[k]) for k in ranks]
    p95s = [np.percentile(mismatch_by_k[k], 95) for k in ranks]
    p99s = [np.percentile(mismatch_by_k[k], 99) for k in ranks]
    maxs = [np.max(mismatch_by_k[k]) for k in ranks]

    mode_label = "Centered" if centered else "Uncentered"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    # 1. Singular Value Energy Spectrum
    ax = axes[0, 0]
    ax.loglog(k_axis, np.maximum(energy_frac, 1e-18), color="navy", lw=1.8)
    ax.axhline(1e-4, color="gray", ls=":", lw=1.0, label="1e-4")
    ax.axhline(1e-6, color="dimgray", ls="--", lw=1.0, label="1e-6")
    ax.axhline(1e-8, color="silver", ls="-.", lw=1.0, label="1e-8")
    ax.set_title(f"Complex128 POD Energy Spectrum ({mode_label})", fontweight="bold")
    ax.set_xlabel("POD Mode Index k")
    ax.set_ylabel("Normalized Energy ($\\sigma_k^2 / \\sum \\sigma_j^2$)")
    ax.set_xlim(1, len(k_axis))
    ax.set_ylim(1e-16, 1.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")

    # 2. Residual Energy Fraction
    ax = axes[0, 1]
    ax.loglog(k_axis, residual_energy, color="teal", lw=1.8)
    ax.axhline(1e-3, color="orange", ls="--", lw=1.0, label="0.1% residual (99.9%)")
    ax.axhline(1e-4, color="crimson", ls=":", lw=1.0, label="0.01% residual (99.99%)")
    ax.axhline(1e-6, color="purple", ls="-.", lw=1.0, label="1e-6 residual")
    ax.set_title("Residual Uncaptured Energy Fraction", fontweight="bold")
    ax.set_xlabel("Truncation Rank k")
    ax.set_ylabel("Relative Residual Energy ($1 - \\sum_{i=1}^k \\lambda_i$)")
    ax.set_xlim(1, len(k_axis))
    ax.set_ylim(1e-14, 1.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left")

    # 3. Mismatch vs Truncation Rank
    ax = axes[1, 0]
    ax.semilogy(ranks, np.maximum(meds, 1e-16), "o-", color="royalblue", lw=2.0, label="Median")
    ax.semilogy(ranks, np.maximum(p95s, 1e-16), "s--", color="darkorange", lw=1.6, label="95th percentile")
    ax.semilogy(ranks, np.maximum(p99s, 1e-16), "^-.", color="crimson", lw=1.6, label="99th percentile")
    ax.semilogy(ranks, np.maximum(maxs, 1e-16), "x:", color="black", lw=1.3, label="Max")
    ax.axhline(1e-3, color="gray", ls="--", lw=1.0, label="Mismatch = 1e-3")
    ax.axhline(1e-4, color="dimgray", ls=":", lw=1.0, label="Mismatch = 1e-4")
    ax.set_xscale("log", base=2)
    ax.set_ylim(1e-16, 2.0)
    ax.set_title("Test-Set Mismatch vs. POD Rank k", fontweight="bold")
    ax.set_xlabel("Truncation Rank k (log2 scale)")
    ax.set_ylabel("Mismatch (1 - Overlap)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)

    # 4. Mismatch CDF for selected ranks
    ax = axes[1, 1]
    cdf_ranks = [r for r in [32, 64, 128, 256, 512, 1024] if r in mismatch_by_k]
    for r in cdf_ranks:
        vals = np.sort(np.maximum(mismatch_by_k[r], 1e-16))
        cdf = np.linspace(0, 1, len(vals), endpoint=True)
        ax.semilogx(vals, cdf, lw=1.8, label=f"k = {r}")
    ax.axvline(1e-3, color="gray", ls="--", lw=1.0)
    ax.axvline(1e-4, color="dimgray", ls=":", lw=1.0)
    ax.set_title("Test-Set Mismatch CDF by Rank", fontweight="bold")
    ax.set_xlabel("Mismatch (1 - Overlap)")
    ax.set_ylabel("Cumulative Fraction of Test Waveforms")
    ax.set_xlim(1e-16, 1.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()


def plot_rank_distribution(mm: np.ndarray, rank: int, output_path: str, centered: bool):
    """Plot log-scale mismatch histogram and summary percentiles for a single POD rank k."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    vals = np.maximum(mm, 1e-16)
    min_exp = np.floor(np.log10(np.min(vals))) - 0.3
    max_exp = max(np.ceil(np.log10(np.max(vals))) + 0.3, min_exp + 2.0)
    bins = np.logspace(min_exp, max_exp, 60)

    med = float(np.median(mm))
    mean_v = float(np.mean(mm))
    p95 = float(np.percentile(mm, 95))
    p99 = float(np.percentile(mm, 99))
    max_v = float(np.max(mm))
    min_v = float(np.min(mm))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=250)
    ax.hist(vals, bins=bins, color="#2b5c8f", edgecolor="#173559", alpha=0.85)
    ax.axvline(med, color="#d62728", ls="-", lw=2.0, label=f"Median: {med:.2e}")
    ax.axvline(p95, color="#ff7f0e", ls="--", lw=1.8, label=f"p95: {p95:.2e}")
    ax.axvline(p99, color="#9467bd", ls="-.", lw=1.8, label=f"p99: {p99:.2e}")
    ax.axvline(max_v, color="black", ls=":", lw=1.5, label=f"Max: {max_v:.2e}")

    ax.set_xscale("log")
    ax.set_xlabel("Mismatch (1 - Overlap)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count (Test Waveforms)", fontsize=11, fontweight="bold")
    mode_label = "Centered" if centered else "Uncentered"
    ax.set_title(
        f"Complex128 POD Test-Set Mismatch Distribution (k = {rank}, {mode_label}, N = {len(mm)})",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    stats_text = (
        f"Min:    {min_v:.2e}\n"
        f"Median: {med:.2e}\n"
        f"Mean:   {mean_v:.2e}\n"
        f"p95:    {p95:.2e}\n"
        f"p99:    {p99:.2e}\n"
        f"Max:    {max_v:.2e}"
    )
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="#2b5c8f", facecolor="#f0f5fa", alpha=0.92),
    )

    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    t0 = time.time()

    with h5py.File(args.dataset, "r") as f:
        waveforms = f[args.key][:]

    assert waveforms.dtype == np.complex128, f"Expected complex128, got {waveforms.dtype}"
    n_samples, length = waveforms.shape
    train_idx, _, test_idx = get_split_indices(n_samples, seed=42)

    X_train = waveforms[train_idx]
    X_test = waveforms[test_idx]
    print(f"Dataset: {args.dataset} [{args.key}] shape={waveforms.shape} dtype={waveforms.dtype}")
    print(f"Split: train={len(X_train)}, test={len(X_test)}, centered={args.centered}")

    t_svd = time.time()
    s, energy_frac, cum_energy, residual_energy, Vh, mean_wf = compute_pod(X_train, centered=args.centered)
    print(f"Complex SVD completed in {time.time() - t_svd:.2f}s (sigma_1={s[0]:.4e}, sigma_min={s[-1]:.4e})")

    print("\nCumulative Energy Thresholds:")
    for thr in [0.90, 0.99, 0.999, 0.9999, 0.99999, 0.999999]:
        k_req = int(np.searchsorted(cum_energy, thr) + 1)
        print(f"  {thr * 100:>8.4f}% -> k = {k_req}")

    ranks = sorted(set([k for k in [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048] if k <= length] + ([args.dist_rank] if args.dist_rank <= length else [])))
    mismatch_by_k = evaluate_mismatches(X_test, Vh, mean_wf, ranks, eps=args.eps)

    print("\nTest-Set Reconstruction Mismatch vs. Rank k:")
    header = f"{'k':>5} | {'Res. Energy':>11} | {'Median':>10} | {'p95':>10} | {'p99':>10} | {'Max':>10} | {'<1e-3 (%)':>9}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for k in ranks:
        mm = mismatch_by_k[k]
        res_e = residual_energy[k - 1]
        pct_1e3 = np.mean(mm < 1e-3) * 100.0
        print(
            f"{k:5d} | {res_e:11.3e} | {np.median(mm):10.3e} | "
            f"{np.percentile(mm, 95):10.3e} | {np.percentile(mm, 99):10.3e} | "
            f"{np.max(mm):10.3e} | {pct_1e3:8.2f}%"
        )
    print("-" * len(header))

    make_plot(energy_frac, residual_energy, mismatch_by_k, args.output_plot, centered=args.centered)
    print(f"\nSaved plot to {args.output_plot}")

    if args.dist_rank in mismatch_by_k:
        dist_path = args.dist_plot or f"runs/pod_k{args.dist_rank}_mismatch_distribution.png"
        mm_k = mismatch_by_k[args.dist_rank]
        plot_rank_distribution(mm_k, args.dist_rank, dist_path, centered=args.centered)
        print(
            f"Rank k={args.dist_rank} stats: min={np.min(mm_k):.3e}, p10={np.percentile(mm_k, 10):.3e}, "
            f"median={np.median(mm_k):.3e}, mean={np.mean(mm_k):.3e}, p90={np.percentile(mm_k, 90):.3e}, "
            f"p95={np.percentile(mm_k, 95):.3e}, p99={np.percentile(mm_k, 99):.3e}, max={np.max(mm_k):.3e}"
        )
        print(f"Saved distribution plot to {dist_path} (total elapsed: {time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
