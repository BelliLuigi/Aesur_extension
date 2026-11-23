#!/usr/bin/env python3
"""
Fit a complex128 POD basis on the training split (80/10/10, seed=42) of
NRSur7dq4v2_dataset_small.hdf and save the rank-k compressed POD dataset
(coefficients, basis, mean, parameters, and split indices) to HDF5.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.join(SCRIPT_DIR, "7dq4CNN")
for path in (SCRIPT_DIR, CNN_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils import TasosOverlapTorch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a rank-k complex128 POD dataset (coefficients + basis) to HDF5."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=256,
        help="POD truncation rank k to save (default: 256)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/NRSur7dq4v2_dataset_small.hdf",
        help="Input HDF5 dataset path (default: datasets/NRSur7dq4v2_dataset_small.hdf)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Waveforms",
        help="Complex128 HDF5 dataset key (default: Waveforms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 path (default: datasets/NRSur7dq4v2_small_pod_k<k>.hdf)",
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
        help="Regularization eps for TasosOverlapTorch verification (default: 1e-30)",
    )
    parser.add_argument(
        "--save_reconstructed",
        action="store_true",
        help="Also store full (N, L) reconstructed waveforms under 'Waveforms_rec'",
    )
    return parser.parse_args()


def get_split_indices(n_samples: int, seed: int = 42):
    idx_all = np.arange(n_samples)
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)
    return (
        np.sort(np.array(train_idx.indices, dtype=np.int64)),
        np.sort(np.array(val_idx.indices, dtype=np.int64)),
        np.sort(np.array(test_idx.indices, dtype=np.int64)),
    )


def compute_pod(X_train: np.ndarray, centered: bool = False):
    """Compute complex POD modes via thin QR + SVD on (N_train, L) complex128 matrix."""
    if centered:
        mean_wf = X_train.mean(axis=0, keepdims=True)
        X_proc = X_train - mean_wf
    else:
        mean_wf = np.zeros((1, X_train.shape[1]), dtype=X_train.dtype)
        X_proc = X_train

    _, R = np.linalg.qr(X_proc, mode="reduced")
    _, s, Vh = np.linalg.svd(R, full_matrices=False)
    energy = s.astype(np.float64) ** 2
    cum_energy = np.cumsum(energy / np.sum(energy))
    residual_energy = np.maximum(1.0 - cum_energy, 1e-18)
    return s, residual_energy, Vh, mean_wf


def main():
    args = parse_args()
    t0 = time.time()

    with h5py.File(args.dataset, "r") as f:
        waveforms = f[args.key][:]
        params = f["Parameters"][:] if "Parameters" in f else None

    assert waveforms.dtype == np.complex128, f"Expected complex128, got {waveforms.dtype}"
    n_samples, length = waveforms.shape
    if not (1 <= args.k <= length):
        raise ValueError(f"Rank k={args.k} must be in [1, {length}]")

    output_path = args.output or f"datasets/NRSur7dq4v2_small_pod_k{args.k}.hdf"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    train_idx, val_idx, test_idx = get_split_indices(n_samples, seed=42)
    X_train = waveforms[train_idx]

    print(f"Dataset: {args.dataset} [{args.key}] shape={waveforms.shape} dtype={waveforms.dtype}")
    print(f"Fitting POD on train={len(X_train)} waveforms (centered={args.centered})...")
    t_svd = time.time()
    s, residual_energy, Vh, mean_wf = compute_pod(X_train, centered=args.centered)
    print(f"SVD completed in {time.time() - t_svd:.2f}s")

    basis_k = Vh[: args.k]  # (k, L) complex128
    coeffs = (waveforms - mean_wf) @ basis_k.conj().T  # (N, k) complex128

    # Verify reconstruction mismatch on held-out test set
    X_test_true = waveforms[test_idx]
    X_test_rec = coeffs[test_idx] @ basis_k + mean_wf
    overlaps = TasosOverlapTorch(torch.from_numpy(X_test_true), torch.from_numpy(X_test_rec), eps=args.eps)
    test_mm = torch.clamp(1.0 - overlaps, min=0.0).numpy()

    res_e = float(residual_energy[args.k - 1])
    med_mm = float(np.median(test_mm))
    p95_mm = float(np.percentile(test_mm, 95))
    max_mm = float(np.max(test_mm))

    with h5py.File(output_path, "w", libver="latest") as f_out:
        if params is not None:
            f_out.create_dataset("Parameters", data=params)
        f_out.create_dataset("POD_coeffs", data=coeffs)
        f_out.create_dataset("POD_basis", data=basis_k)
        f_out.create_dataset("POD_mean", data=mean_wf)
        f_out.create_dataset("singular_values", data=s[: args.k])

        if args.save_reconstructed:
            X_all_rec = coeffs @ basis_k + mean_wf
            f_out.create_dataset("Waveforms_rec", data=X_all_rec)

        g_splits = f_out.create_group("Splits")
        g_splits.create_dataset("train_idx", data=train_idx)
        g_splits.create_dataset("val_idx", data=val_idx)
        g_splits.create_dataset("test_idx", data=test_idx)

        f_out.attrs["source_dataset"] = args.dataset
        f_out.attrs["source_key"] = args.key
        f_out.attrs["k"] = args.k
        f_out.attrs["centered"] = bool(args.centered)
        f_out.attrs["split_seed"] = 42
        f_out.attrs["split_ratios"] = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        f_out.attrs["residual_energy"] = res_e
        f_out.attrs["test_median_mismatch"] = med_mm
        f_out.attrs["test_p95_mismatch"] = p95_mm
        f_out.attrs["test_max_mismatch"] = max_mm

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")
    print(
        f"k={args.k} | res_energy={res_e:.3e} | "
        f"test mismatch median={med_mm:.3e}, p95={p95_mm:.3e}, max={max_mm:.3e} "
        f"(elapsed: {time.time() - t0:.2f}s)"
    )


if __name__ == "__main__":
    main()
