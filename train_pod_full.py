#!/usr/bin/env python3
"""
Full 3-stage training pipeline for the complex128 POD + Autoencoder + Mapper surrogate:
  Stage 1: PODAutoencoder (512 -> 256 -> 128 -> 64 -> 128 -> 256 -> 512), 1200 epochs, cosine 1e-3 -> 1e-7
  Stage 2: PODMapper (7 -> 16 -> 32 -> 64 -> 64) + frozen PODDecoder, 600 epochs, cosine 1e-3 -> 1e-6
  Stage 3: PODFull (PODMapper + unfrozen PODDecoder), 600 epochs, cosine 5e-5 -> 5e-7
  Eval:    Computes and plots test-set waveform mismatch (TasosOverlapTorch) after EACH stage.
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
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.join(SCRIPT_DIR, "7dq4CNN")
for path in (SCRIPT_DIR, CNN_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from modelsPOD import (
    PODAutoencoder,
    PODDecoder,
    PODFullModel,
    PODMapper,
    PODReconstructor,
    coeffs_complex_to_real,
    count_parameters,
)
from utils import TasosOverlapTorch


class MapperFrozenDecoder(nn.Module):
    """Stage 2/3 wrapper: params (B, 7) -> PODMapper -> PODDecoder -> c_hat (B, 2*k)."""

    def __init__(self, mapper: PODMapper, decoder: PODDecoder):
        super().__init__()
        self.mapper = mapper
        self.decoder = decoder

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.mapper(params))


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3-stage POD + AE + Mapper pipeline with SOAP and per-stage eval.")
    parser.add_argument("--title", type=str, default=None, help="Run prefix for checkpoints and plots")
    parser.add_argument(
        "--pod_hdf",
        type=str,
        default="datasets/NRSur7dq4v2_podk256_dataset.hdf",
        help="Input POD HDF5 created by pod_complex_save.py",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size (default: 256)")

    # Stage 1: PODAutoencoder
    parser.add_argument("--ae_epochs", type=int, default=1200, help="Stage 1 AE epochs (default: 1200)")
    parser.add_argument("--lr_ae", type=float, default=1e-3, help="Stage 1 start LR (default: 1e-3)")
    parser.add_argument("--min_lr_ae", type=float, default=1e-7, help="Stage 1 cosine min LR (default: 1e-7)")

    # Stage 2: PODMapper + frozen PODDecoder
    parser.add_argument("--mapper_epochs", type=int, default=600, help="Stage 2 Mapper epochs (default: 600)")
    parser.add_argument("--lr_mapper", type=float, default=1e-3, help="Stage 2 start LR (default: 1e-3)")
    parser.add_argument("--min_lr_mapper", type=float, default=1e-6, help="Stage 2 cosine min LR (default: 1e-6)")

    # Stage 3: Joint PODMapper + unfrozen PODDecoder
    parser.add_argument("--joint_epochs", type=int, default=600, help="Stage 3 joint epochs (default: 600)")
    parser.add_argument("--lr_joint", type=float, default=5e-5, help="Stage 3 start LR (default: 5e-5)")
    parser.add_argument("--min_lr_joint", type=float, default=5e-7, help="Stage 3 cosine min LR (default: 5e-7)")

    parser.add_argument(
        "--loss",
        type=str,
        choices=["rel_l2", "mse", "l1"],
        default="rel_l2",
        help="Coefficient reconstruction loss (default: rel_l2)",
    )
    parser.add_argument(
        "--f64",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train in float64 precision (default: True; pass --no-f64 for float32)",
    )
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--eps", type=float, default=1e-30, help="eps for TasosOverlapTorch evaluation (default: 1e-30)")
    return parser.parse_args()


def make_loss_fn(loss_name: str):
    if loss_name == "rel_l2":
        def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            num = torch.sum((pred - target) ** 2, dim=-1)
            den = torch.sum(target ** 2, dim=-1) + 1e-12
            return torch.mean(num / den)
        return rel_l2
    if loss_name == "mse":
        return nn.MSELoss()
    return nn.L1Loss()


def make_soap_and_cosine(params, lr: float, min_lr: float, total_steps: int, f64: bool, warmup_steps: int = 0):
    soap_eps = 1e-14 if f64 else 1e-8
    optimizer = opt_add.SOAP(
        params,
        lr=lr,
        betas=(0.95, 0.95),
        shampoo_beta=0.99,
        weight_decay=0.0,
        precondition_frequency=10,
        merge_dims=False,
        eps=soap_eps,
    )
    if warmup_steps > 0 and total_steps > warmup_steps:
        decay_steps = total_steps - warmup_steps
        warmup = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=min_lr)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=min_lr)
    return optimizer, scheduler


def run_stage(
    name: str,
    model: nn.Module,
    trainable_params,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    epochs: int,
    lr: float,
    min_lr: float,
    f64: bool,
    clip_norm: float,
    device: torch.device,
    warmup_steps: int = 0,
):
    if epochs <= 0:
        return [], [], None

    total_steps = epochs * len(train_loader)
    optimizer, scheduler = make_soap_and_cosine(
        trainable_params,
        lr=lr,
        min_lr=min_lr,
        total_steps=total_steps,
        f64=f64,
        warmup_steps=warmup_steps,
    )
    best_val = float("inf")
    best_state = None
    train_hist, val_hist = [], []

    print(
        f"\n--- {name} | epochs={epochs} | lr={lr:.1e} -> {min_lr:.1e} (cosine) | "
        f"trainable_params={sum(p.numel() for p in trainable_params):,} ---"
    )
    t0 = time.time()
    log_every = max(1, epochs // 12)

    for epoch in range(1, epochs + 1):
        model.train(True)
        tr_sum = 0.0
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
            optimizer.step()
            scheduler.step()
            tr_sum += float(loss.item()) * len(x_b)
        tr_loss = tr_sum / len(train_loader.dataset)

        model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                va_sum += float(loss_fn(model(x_b), y_b).item()) * len(x_b)
        va_loss = va_sum / len(val_loader.dataset)

        train_hist.append(tr_loss)
        val_hist.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}/{epochs} | train={tr_loss:.3e} | val={va_loss:.3e} | best={best_val:.3e} | lr={cur_lr:.2e}")

    print(f"--- {name} finished in {time.time() - t0:.1f}s (best val={best_val:.3e}) ---")
    return train_hist, val_hist, best_state


def compute_mismatch(h_true: torch.Tensor, h_pred: torch.Tensor, eps: float) -> np.ndarray:
    overlaps = TasosOverlapTorch(h_true, h_pred, eps=eps)
    return torch.clamp(1.0 - overlaps, min=0.0).cpu().numpy()


def print_mismatch_stats(stage_label: str, mm: np.ndarray):
    pct_1e3 = float(np.mean(mm < 1e-3) * 100.0)
    pct_1e4 = float(np.mean(mm < 1e-4) * 100.0)
    print(
        f"  [Eval] {stage_label:<28} | med={np.median(mm):.3e} | mean={np.mean(mm):.3e} | "
        f"p95={np.percentile(mm, 95):.3e} | p99={np.percentile(mm, 99):.3e} | max={np.max(mm):.3e} | "
        f"<1e-3={pct_1e3:.1f}% | <1e-4={pct_1e4:.1f}%"
    )


def plot_single_stage_mismatch(mm: np.ndarray, stage_title: str, out_path: str, color: str = "#2b5c8f"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vals = np.maximum(mm, 1e-16)
    min_exp = np.floor(np.log10(np.min(vals))) - 0.3
    max_exp = max(np.ceil(np.log10(np.max(vals))) + 0.3, min_exp + 2.0)
    bins = np.logspace(min_exp, max_exp, 60)

    med = float(np.median(mm))
    mean_v = float(np.mean(mm))
    p95 = float(np.percentile(mm, 95))
    p99 = float(np.percentile(mm, 99))
    max_v = float(np.max(mm))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=220)
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
    fig.savefig(out_path)
    plt.close(fig)


def plot_curves(histories: dict, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    active = [(k, v) for k, v in histories.items() if len(v[0]) > 0]
    if not active:
        return
    fig, axes = plt.subplots(1, len(active), figsize=(5.5 * len(active), 4.5), dpi=200)
    if len(active) == 1:
        axes = [axes]
    for ax, (stage_name, (tr_h, va_h)) in zip(axes, active):
        ep = np.arange(1, len(tr_h) + 1)
        ax.semilogy(ep, tr_h, label="Train", lw=1.6)
        ax.semilogy(ep, va_h, label="Val", lw=1.6)
        ax.set_title(stage_name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
    fig.suptitle(f"POD Pipeline Training Curves — {title}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mismatch_comparison(mm_dict: dict[str, np.ndarray], out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    all_vals = np.concatenate([np.maximum(v, 1e-16) for v in mm_dict.values()])
    min_exp = np.floor(np.log10(np.min(all_vals))) - 0.3
    max_exp = max(np.ceil(np.log10(np.max(all_vals))) + 0.3, min_exp + 2.0)
    bins = np.logspace(min_exp, max_exp, 65)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=220)
    colors = {
        "POD Floor (k=256)": "#444444",
        "Stage 1: AE": "#1f77b4",
        "Stage 2: Mapper (Frozen Dec)": "#ff7f0e",
        "Stage 3: Full (Joint)": "#d62728",
    }
    for label, mm in mm_dict.items():
        med = float(np.median(mm))
        p95 = float(np.percentile(mm, 95))
        c = colors.get(label, "#2ca02c")
        ax.hist(
            np.maximum(mm, 1e-16),
            bins=bins,
            histtype="step",
            lw=1.9,
            color=c,
            label=f"{label} (med={med:.2e}, p95={p95:.2e})",
        )
        ax.axvline(med, color=c, ls=":", lw=1.2)

    ax.set_xscale("log")
    ax.set_xlabel("Test-Set Waveform Mismatch (1 - Overlap)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(f"Test-Set Mismatch by Stage — {title}", fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    dtype = torch.float64 if args.f64 else torch.float32
    if args.f64:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    title = args.title or f"POD_k256_SOAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    with h5py.File(args.pod_hdf, "r") as f:
        k = int(f.attrs["k"])
        source_dataset = str(f.attrs["source_dataset"])
        source_key = str(f.attrs["source_key"])
        coeffs_c = f["POD_coeffs"][:]
        params_np = f["Parameters"][:]
        train_idx = f["Splits/train_idx"][:]
        val_idx = f["Splits/val_idx"][:]
        test_idx = f["Splits/test_idx"][:]

    with h5py.File(source_dataset, "r") as f_raw:
        h_test_true = torch.from_numpy(f_raw[source_key][test_idx]).to(device)

    coeffs_t = coeffs_complex_to_real(torch.from_numpy(coeffs_c)).to(dtype=dtype)
    params_raw = torch.from_numpy(params_np).to(dtype=dtype)

    # Standardize 7D input parameters strictly on train split
    p_mu = params_raw[train_idx].mean(dim=0, keepdim=True)
    p_sd = params_raw[train_idx].std(dim=0, keepdim=True).clamp(min=1e-8)
    params_norm = (params_raw - p_mu) / p_sd

    c_test = coeffs_t[test_idx].to(device)
    p_test = params_norm[test_idx].to(device)

    print(f"=== Run: {title} | device={device} | dtype={dtype} | k={k} | loss={args.loss} ===")
    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    ae = PODAutoencoder(in_dim=2 * k, hidden_dims=(256, 128), latent_dim=64, dtype=dtype).to(device)
    mapper = PODMapper(in_params=params_norm.shape[1], hidden_dims=(16, 32, 64), latent_dim=64, dtype=dtype).to(device)
    recon = PODReconstructor.from_hdf5(args.pod_hdf, trainable=False, dtype=dtype).to(device)
    print(
        f"Params: AE={count_parameters(ae):,}, Mapper={count_parameters(mapper):,}, "
        f"Decoder={count_parameters(ae.decoder):,}, Reconstructor={count_parameters(recon):,} (frozen)"
    )

    meta = {
        "pod_hdf": args.pod_hdf,
        "k": k,
        "f64": args.f64,
        "param_mu": p_mu.cpu(),
        "param_sd": p_sd.cpu(),
    }

    loss_fn = make_loss_fn(args.loss)
    stage_mismatches: dict[str, np.ndarray] = {}

    with torch.no_grad():
        mm_floor = compute_mismatch(h_test_true, recon(c_test), eps=args.eps)
    stage_mismatches[f"POD Floor (k={k})"] = mm_floor
    print_mismatch_stats(f"POD Floor (k={k})", mm_floor)

    # ------------------------------------------------------------------
    # Stage 1: PODAutoencoder (1200 epochs, cosine 1e-3 -> 1e-7)
    # ------------------------------------------------------------------
    ae_train_loader = DataLoader(TensorDataset(coeffs_t[train_idx], coeffs_t[train_idx]), batch_size=args.batch_size, shuffle=True)
    ae_val_loader = DataLoader(TensorDataset(coeffs_t[val_idx], coeffs_t[val_idx]), batch_size=1024, shuffle=False)

    tr_ae, va_ae, best_ae_state = run_stage(
        "Stage 1: PODAutoencoder",
        ae,
        list(ae.parameters()),
        ae_train_loader,
        ae_val_loader,
        loss_fn,
        args.ae_epochs,
        args.lr_ae,
        args.min_lr_ae,
        args.f64,
        args.clip_norm,
        device,
        warmup_steps=min(200, max(1, (args.ae_epochs * len(ae_train_loader)) // 20)),
    )
    if best_ae_state is not None:
        ae.load_state_dict(best_ae_state)

    ae.eval()
    with torch.no_grad():
        mm_stage1 = compute_mismatch(h_test_true, recon(ae(c_test)), eps=args.eps)
    stage_mismatches["Stage 1: AE"] = mm_stage1
    print_mismatch_stats("Stage 1: PODAutoencoder", mm_stage1)
    torch.save({**meta, "model_state_dict": ae.state_dict()}, f"model_weights/{title}_stage1_ae_best.pt")
    plot_single_stage_mismatch(
        mm_stage1,
        f"Stage 1 Mismatch: PODAutoencoder ({title}, k={k})",
        f"runs/{title}_stage1_ae_mismatch.png",
        color="#1f77b4",
    )

    # ------------------------------------------------------------------
    # Stage 2: PODMapper + Frozen PODDecoder (600 epochs, cosine 1e-3 -> 1e-6)
    # ------------------------------------------------------------------
    for p in ae.decoder.parameters():
        p.requires_grad_(False)
    stage2_model = MapperFrozenDecoder(mapper, ae.decoder).to(device)

    p_train_loader = DataLoader(TensorDataset(params_norm[train_idx], coeffs_t[train_idx]), batch_size=args.batch_size, shuffle=True)
    p_val_loader = DataLoader(TensorDataset(params_norm[val_idx], coeffs_t[val_idx]), batch_size=1024, shuffle=False)

    tr_map, va_map, best_stage2_state = run_stage(
        "Stage 2: PODMapper + Frozen PODDecoder",
        stage2_model,
        list(mapper.parameters()),
        p_train_loader,
        p_val_loader,
        loss_fn,
        args.mapper_epochs,
        args.lr_mapper,
        args.min_lr_mapper,
        args.f64,
        args.clip_norm,
        device,
        warmup_steps=min(100, max(1, (args.mapper_epochs * len(p_train_loader)) // 20)),
    )
    if best_stage2_state is not None:
        stage2_model.load_state_dict(best_stage2_state)

    stage2_model.eval()
    with torch.no_grad():
        mm_stage2 = compute_mismatch(h_test_true, recon(stage2_model(p_test)), eps=args.eps)
    stage_mismatches["Stage 2: Mapper (Frozen Dec)"] = mm_stage2
    print_mismatch_stats("Stage 2: Mapper + Frozen Dec", mm_stage2)
    torch.save({**meta, "model_state_dict": stage2_model.state_dict()}, f"model_weights/{title}_stage2_mapper_best.pt")
    plot_single_stage_mismatch(
        mm_stage2,
        f"Stage 2 Mismatch: PODMapper + Frozen Decoder ({title})",
        f"runs/{title}_stage2_mapper_mismatch.png",
        color="#ff7f0e",
    )

    # ------------------------------------------------------------------
    # Stage 3: PODFull (Mapper + Unfrozen Decoder, 600 epochs, cosine 5e-5 -> 5e-7)
    # ------------------------------------------------------------------
    for p in ae.decoder.parameters():
        p.requires_grad_(True)
    stage3_params = list(mapper.parameters()) + list(ae.decoder.parameters())

    tr_joint, va_joint, best_stage3_state = run_stage(
        "Stage 3: PODFull (Mapper + Decoder)",
        stage2_model,
        stage3_params,
        p_train_loader,
        p_val_loader,
        loss_fn,
        args.joint_epochs,
        args.lr_joint,
        args.min_lr_joint,
        args.f64,
        args.clip_norm,
        device,
        warmup_steps=0,
    )
    if best_stage3_state is not None:
        stage2_model.load_state_dict(best_stage3_state)

    full_model = PODFullModel(mapper, ae.decoder, recon).to(device)
    full_model.eval()
    with torch.no_grad():
        mm_stage3 = compute_mismatch(h_test_true, full_model(p_test), eps=args.eps)
    stage_mismatches["Stage 3: Full (Joint)"] = mm_stage3
    print_mismatch_stats("Stage 3: PODFull (Joint)", mm_stage3)
    torch.save({**meta, "model_state_dict": full_model.state_dict()}, f"model_weights/{title}_stage3_full_best.pt")
    plot_single_stage_mismatch(
        mm_stage3,
        f"Stage 3 Mismatch: PODFull Joint Mapper + Decoder ({title})",
        f"runs/{title}_stage3_full_mismatch.png",
        color="#d62728",
    )

    # Save combined plots + raw mismatch arrays
    curves_path = f"runs/{title}_training_curves.png"
    cmp_path = f"runs/{title}_mismatch_comparison.png"
    npz_path = f"runs/{title}_mismatches.npz"

    plot_curves(
        {
            "Stage 1: AE": (tr_ae, va_ae),
            "Stage 2: Mapper (Frozen Dec)": (tr_map, va_map),
            "Stage 3: PODFull (Joint)": (tr_joint, va_joint),
        },
        curves_path,
        title,
    )
    plot_mismatch_comparison(stage_mismatches, cmp_path, title)
    np.savez_compressed(
        npz_path,
        pod_floor=mm_floor,
        stage1_ae=mm_stage1,
        stage2_mapper=mm_stage2,
        stage3_full=mm_stage3,
    )

    print("\nSaved per-stage checkpoints:")
    print(f"  - model_weights/{title}_stage1_ae_best.pt")
    print(f"  - model_weights/{title}_stage2_mapper_best.pt")
    print(f"  - model_weights/{title}_stage3_full_best.pt")
    print("Saved plots & metrics:")
    print(f"  - runs/{title}_stage1_ae_mismatch.png")
    print(f"  - runs/{title}_stage2_mapper_mismatch.png")
    print(f"  - runs/{title}_stage3_full_mismatch.png")
    print(f"  - {cmp_path}")
    print(f"  - {curves_path}")
    print(f"  - {npz_path}")


if __name__ == "__main__":
    main()
