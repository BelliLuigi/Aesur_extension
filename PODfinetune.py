#!/usr/bin/env python3
"""
K-Fold Cross-Validation SXS Finetuning for PODFullModel:
  params (B, 7) -> PODMapper -> PODDecoder -> PODReconstructor -> waveform (B, 2048)

Loads a pretrained Stage 3 checkpoint from train_pod_full.py, holds out a 10% FPS
test set from datasets/sxs_waveforms.h5 (7dq4 domain), and fine-tunes across K folds
using SOAP + CosineAnnealingLR with low per-component learning rates.
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
import numpy as np
import pytorch_optimizer as opt_add
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.join(SCRIPT_DIR, "7dq4CNN")
for path in (SCRIPT_DIR, CNN_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from modelsPOD import PODDecoder, PODFullModel, PODMapper, PODReconstructor, count_parameters
from utils import TasosOverlapTorch


def parse_args():
    parser = argparse.ArgumentParser(description="K-Fold SXS finetuning for PODFullModel with SOAP + CosineAnnealing.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 3 checkpoint (*_stage3_full_best.pt)")
    parser.add_argument("--sxs_path", type=str, default="datasets/sxs_waveforms.h5", help="Path to sxs_waveforms.h5")
    parser.add_argument("--title", type=str, default=None, help="Run prefix for checkpoints and plots")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5; set 1 for single split)")
    parser.add_argument("--epochs", type=int, default=300, help="Finetuning epochs per fold (default: 300)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")

    # Low learning rates with cosine annealing
    parser.add_argument("--lr_mapper", type=float, default=5e-5, help="Initial LR for PODMapper (default: 5e-5)")
    parser.add_argument("--lr_decoder", type=float, default=1e-5, help="Initial LR for PODDecoder (default: 1e-5, 0 to freeze)")
    parser.add_argument("--lr_recon", type=float, default=1e-6, help="Initial LR for PODReconstructor (default: 1e-6, 0 to freeze)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="Cosine eta_min = lr * min_lr_ratio (default: 0.01)")

    parser.add_argument(
        "--loss",
        type=str,
        choices=["wave_rel_l2", "wave_mse", "mismatch"],
        default="wave_rel_l2",
        help="Waveform finetuning loss (default: wave_rel_l2)",
    )
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--eps", type=float, default=1e-30, help="eps for TasosOverlapTorch (default: 1e-30)")
    return parser.parse_args()


def farthest_point_sampling(pts: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Select n_samples spanning the normalized 7D parameter space."""
    p_norm = (pts - pts.min(dim=0).values) / (pts.max(dim=0).values - pts.min(dim=0).values + 1e-8)
    selected = [0]
    dists = torch.cdist(p_norm, p_norm[0:1]).squeeze(1)
    for _ in range(1, n_samples):
        next_idx = int(torch.argmax(dists).item())
        selected.append(next_idx)
        dists = torch.minimum(dists, torch.cdist(p_norm, p_norm[next_idx : next_idx + 1]).squeeze(1))
    return torch.tensor(selected, dtype=torch.long)


def load_sxs_7dq4(sxs_path: str):
    """Load and filter SXS (2,2) waveforms matching the NRSur7dq4v2 parameter domain."""
    waves, params, names = [], [], []
    with h5py.File(sxs_path, "r") as f:
        for sim_name in f.keys():
            g = f[sim_name]
            if "waveform" not in g or "(2,2)" not in g["waveform"]:
                continue
            wf = g["waveform/(2,2)"][:]
            if len(wf) != 2048:
                continue
            q = float(g["metadata/mass_ratio"][()])
            s1 = g["metadata/dimensionless_spin1"][:]
            s2 = g["metadata/dimensionless_spin2"][:]
            ecc = float(g["metadata/eccentricity"][()])
            if (
                1.0 <= q <= 4.0
                and np.linalg.norm(s1) <= 0.81
                and np.linalg.norm(s2) <= 0.81
                and abs(ecc) < 1e-2
                and np.sum(np.abs(wf)) < 1000.0
                and abs(wf[-1]) < 0.5
            ):
                waves.append(wf)
                params.append([q, float(s1[0]), float(s1[1]), float(s1[2]), float(s2[0]), float(s2[1]), float(s2[2])])
                names.append(sim_name)

    return np.stack(waves), np.stack(params), np.array(names)


def build_full_model_from_ckpt(ckpt: dict, device: torch.device):
    """Reconstruct PODFullModel and load Stage 3 weights."""
    k = int(ckpt["k"])
    f64 = bool(ckpt.get("f64", True))
    dtype = torch.float64 if f64 else torch.float32
    pod_hdf = str(ckpt["pod_hdf"])

    mapper_hidden = tuple(ckpt.get("mapper_hidden", (16, 32, 64)))  # old checkpoints used (16, 32, 64)
    mapper = PODMapper(in_params=7, hidden_dims=mapper_hidden, latent_dim=64, dtype=dtype)
    decoder = PODDecoder(latent_dim=64, hidden_dims=(128, 256), out_dim=2 * k, dtype=dtype)
    recon = PODReconstructor.from_hdf5(pod_hdf, trainable=False, dtype=dtype)
    model = PODFullModel(mapper, decoder, recon).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, dtype, f64


def make_loss_fn(loss_name: str, eps: float):
    if loss_name == "wave_rel_l2":
        def wave_rel_l2(h_pred: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
            num = torch.sum((h_pred.real - h_true.real) ** 2 + (h_pred.imag - h_true.imag) ** 2, dim=-1)
            den = torch.sum(h_true.real ** 2 + h_true.imag ** 2, dim=-1) + 1e-12
            return torch.mean(num / den)
        return wave_rel_l2
    if loss_name == "wave_mse":
        def wave_mse(h_pred: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
            return torch.mean((h_pred.real - h_true.real) ** 2 + (h_pred.imag - h_true.imag) ** 2)
        return wave_mse
    def mismatch_loss(h_pred: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
        return torch.mean(1.0 - TasosOverlapTorch(h_true, h_pred, eps=eps))
    return mismatch_loss


def eval_mismatch(model: PODFullModel, p_tensor: torch.Tensor, h_true: torch.Tensor, eps: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        h_pred = model(p_tensor)
        overlaps = TasosOverlapTorch(h_true, h_pred, eps=eps)
        return torch.clamp(1.0 - overlaps, min=0.0).cpu().numpy()


def plot_cv_curves(fold_histories: list[tuple[list[float], list[float]]], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=200)
    for i, (tr_h, va_h) in enumerate(fold_histories, start=1):
        ep = np.arange(1, len(tr_h) + 1)
        ax.semilogy(ep, tr_h, alpha=0.45, lw=1.2, label=f"Fold {i} Train" if i == 1 else None, color="#1f77b4")
        ax.semilogy(ep, va_h, alpha=0.80, lw=1.6, label=f"Fold {i} Val")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title(f"SXS K-Fold Finetuning Curves — {title}", fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sxs_mismatch(mm_before: np.ndarray, mm_after_folds: list[np.ndarray], out_path: str, title: str):
    mm_best = mm_after_folds[int(np.argmin([np.median(m) for m in mm_after_folds]))]
    mm_all_folds = np.concatenate(mm_after_folds)
    all_vals = np.maximum(np.concatenate([mm_before, mm_best, mm_all_folds]), 1e-16)
    min_exp = np.floor(np.log10(np.min(all_vals))) - 0.3
    max_exp = max(np.ceil(np.log10(np.max(all_vals))) + 0.3, min_exp + 2.0)
    bins = np.logspace(min_exp, max_exp, 40)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=220)
    for label, mm, color in [
        ("Before Finetune (Zero-Shot)", mm_before, "#444444"),
        ("After Finetune (All Folds)", mm_all_folds, "#1f77b4"),
        ("After Finetune (Best Fold)", mm_best, "#d62728"),
    ]:
        med = float(np.median(mm))
        p95 = float(np.percentile(mm, 95))
        ax.hist(np.maximum(mm, 1e-16), bins=bins, histtype="step", lw=1.9, color=color, density=True,
                label=f"{label} (med={med:.2e}, p95={p95:.2e})")
        ax.axvline(med, color=color, ls=":", lw=1.3)

    ax.set_xscale("log")
    ax.set_xlabel("Held-Out SXS Test Mismatch (1 - Overlap)", fontweight="bold")
    ax.set_ylabel("Normalized Density", fontweight="bold")
    ax.set_title(f"SXS Held-Out Test Mismatch — {title}", fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    f64 = bool(ckpt.get("f64", True))
    dtype = torch.float64 if f64 else torch.float32
    wf_dtype = torch.complex128 if f64 else torch.complex64

    if f64:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    title = args.title or f"POD_SXS_KF{args.folds}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Load SXS data and normalize parameters with pretrained NRSur7dq4v2 stats
    waves_np, params_np, names_np = load_sxs_7dq4(args.sxs_path)
    p_mu = ckpt["param_mu"].to(dtype=dtype)
    p_sd = ckpt["param_sd"].to(dtype=dtype)

    params_raw = torch.from_numpy(params_np).to(dtype=dtype)
    params_norm = (params_raw - p_mu) / p_sd
    waves_t = torch.from_numpy(waves_np).to(dtype=wf_dtype)

    print(f"=== SXS Finetune: {title} | N={len(waves_t)} | folds={args.folds} | epochs={args.epochs} | dtype={dtype} ===")

    # Held-out 10% test set via FPS in 7D parameter space
    n_test = max(1, int(0.10 * len(waves_t)))
    test_idx = farthest_point_sampling(params_raw, n_test)
    pool_mask = torch.ones(len(waves_t), dtype=torch.bool)
    pool_mask[test_idx] = False
    pool_idx = torch.arange(len(waves_t))[pool_mask]

    p_test = params_norm[test_idx].to(device)
    h_test = waves_t[test_idx].to(device)
    print(f"Held-out FPS test set: {len(test_idx)} | K-Fold pool: {len(pool_idx)}")

    # Zero-shot evaluation before finetuning
    base_model, _, _ = build_full_model_from_ckpt(ckpt, device)
    mm_before = eval_mismatch(base_model, p_test, h_test, eps=args.eps)
    print(f"[Before Finetune] test med={np.median(mm_before):.3e} | p95={np.percentile(mm_before, 95):.3e} | max={np.max(mm_before):.3e}")
    del base_model

    # Build K-fold splits
    perm = pool_idx[torch.randperm(len(pool_idx), generator=torch.Generator().manual_seed(42))]
    if args.folds == 1:
        n_val = max(1, int(0.15 * len(perm)))
        fold_splits = [(perm[n_val:], perm[:n_val])]
    else:
        chunks = torch.chunk(perm, chunks=args.folds)
        fold_splits = [(torch.cat([chunks[i] for i in range(args.folds) if i != f]), chunks[f]) for f in range(args.folds)]

    full_dataset = TensorDataset(params_norm, waves_t)
    loss_fn = make_loss_fn(args.loss, eps=args.eps)
    soap_eps = 1e-12 if f64 else 1e-8

    fold_histories = []
    mm_after_folds = []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits, start=1):
        model, _, _ = build_full_model_from_ckpt(ckpt, device)

        # Configure per-component LRs
        param_groups = []
        trainable_params = []

        model.mapper.requires_grad_(args.lr_mapper > 0)
        if args.lr_mapper > 0:
            p_m = list(model.mapper.parameters())
            param_groups.append({"params": p_m, "lr": args.lr_mapper})
            trainable_params.extend(p_m)

        model.decoder.requires_grad_(args.lr_decoder > 0)
        if args.lr_decoder > 0:
            p_d = list(model.decoder.parameters())
            param_groups.append({"params": p_d, "lr": args.lr_decoder})
            trainable_params.extend(p_d)

        if args.lr_recon > 0:
            model.reconstructor.unfreeze()
            p_r = list(model.reconstructor.parameters())
            param_groups.append({"params": p_r, "lr": args.lr_recon})
            trainable_params.extend(p_r)
        else:
            model.reconstructor.freeze()

        print(
            f"\n--- Fold {fold_idx}/{args.folds} | train={len(tr_idx)} val={len(va_idx)} | "
            f"trainable={count_parameters(model, trainable_only=True):,} "
            f"(lr: map={args.lr_mapper:.1e}, dec={args.lr_decoder:.1e}, rec={args.lr_recon:.1e}) ---"
        )

        tr_loader = DataLoader(Subset(full_dataset, tr_idx), batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(Subset(full_dataset, va_idx), batch_size=len(va_idx), shuffle=False)

        optimizer = opt_add.SOAP(
            param_groups,
            betas=(0.95, 0.95),
            shampoo_beta=0.99,
            weight_decay=0.0,
            precondition_frequency=10,
            merge_dims=False,
            eps=soap_eps,
        )
        total_steps = args.epochs * len(tr_loader)
        eta_min = min(pg["lr"] for pg in param_groups) * args.min_lr_ratio
        # LambdaLR cosine per group so each group decays from its own LR to lr * min_lr_ratio
        cosine_fn = lambda step: args.min_lr_ratio + 0.5 * (1.0 - args.min_lr_ratio) * (1.0 + np.cos(np.pi * step / max(total_steps, 1)))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[cosine_fn] * len(param_groups))

        best_val = float("inf")
        best_state = None
        tr_hist, va_hist = [], []
        log_every = max(1, args.epochs // 10)
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train(True)
            tr_sum = 0.0
            for x_b, y_b in tr_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x_b), y_b)
                loss.backward()
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.clip_norm)
                optimizer.step()
                scheduler.step()
                tr_sum += float(loss.item()) * len(x_b)
            tr_loss = tr_sum / len(tr_idx)

            model.eval()
            va_sum = 0.0
            with torch.no_grad():
                for x_b, y_b in va_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    va_sum += float(loss_fn(model(x_b), y_b).item()) * len(x_b)
            va_loss = va_sum / len(va_idx)

            tr_hist.append(tr_loss)
            va_hist.append(va_loss)

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}

            if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
                print(
                    f"  Epoch {epoch:4d}/{args.epochs} | train={tr_loss:.3e} | val={va_loss:.3e} "
                    f"| best={best_val:.3e} | lr_map={optimizer.param_groups[0]['lr']:.2e}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        mm_fold = eval_mismatch(model, p_test, h_test, eps=args.eps)
        mm_after_folds.append(mm_fold)
        fold_histories.append((tr_hist, va_hist))

        fold_ckpt = f"model_weights/{title}_fold{fold_idx}_best.pt"
        torch.save(
            {
                "pod_hdf": ckpt["pod_hdf"],
                "k": ckpt["k"],
                "f64": f64,
                "param_mu": p_mu.cpu(),
                "param_sd": p_sd.cpu(),
                "fold": fold_idx,
                "test_idx": test_idx.cpu(),
                "model_state_dict": model.state_dict(),
            },
            fold_ckpt,
        )
        print(
            f"  [Fold {fold_idx} Eval ({time.time() - t0:.1f}s)] test med={np.median(mm_fold):.3e} | "
            f"p95={np.percentile(mm_fold, 95):.3e} | max={np.max(mm_fold):.3e} | saved={fold_ckpt}"
        )

    # Aggregate CV summary & plots
    meds = [float(np.median(m)) for m in mm_after_folds]
    p95s = [float(np.percentile(m, 95)) for m in mm_after_folds]
    print(f"\n=== K-Fold Summary ({args.folds} folds) ===")
    print(f"  Before finetune: med={np.median(mm_before):.3e} | p95={np.percentile(mm_before, 95):.3e}")
    print(f"  After finetune:  med={np.mean(meds):.3e} +/- {np.std(meds):.2e} | p95={np.mean(p95s):.3e}")

    curves_path = f"runs/{title}_cv_curves.png"
    mm_path = f"runs/{title}_sxs_mismatch.png"
    npz_path = f"runs/{title}_sxs_mismatches.npz"
    plot_cv_curves(fold_histories, curves_path, title)
    plot_sxs_mismatch(mm_before, mm_after_folds, mm_path, title)
    np.savez_compressed(npz_path, mm_before=mm_before, **{f"fold{i+1}": m for i, m in enumerate(mm_after_folds)})
    print(f"Saved plots & metrics: {curves_path}, {mm_path}, {npz_path}")


if __name__ == "__main__":
    main()
