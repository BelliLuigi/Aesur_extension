#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

# Paths
ckpt_path = "model_weights/Monolithlogfreq_logfreq_checkpoint.pt"
norm_path = "model_weights/Monolithlogfreq_logfreq_norm.npz"
data_path = "datasets/NRSur7dq4v2_dataset.hdf"

from models import AutoencoderMono
from utils import TasosOverlapTorch

def main():
    print("=" * 70)
    print("COMPUTING MISMATCH DISTRIBUTION: MONOLITH LOG-FREQ (TRUE AMPLITUDE)")
    print("=" * 70)

    # 1. Load Model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AutoencoderMono()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {ckpt_path} (Epoch: {ckpt.get('epoch')}, vloss: {ckpt.get('best_vloss'):.6e})")

    # 2. Load Normalization
    norm_data = np.load(norm_path)
    mu, sd = norm_data["mu"], norm_data["sd"]
    omega_floor = float(norm_data.get("omega_floor", 1e-6))
    clamp_target = 4.0

    # 3. Get Test Set Indices (matching train_monolith_logfreq.py seed=42)
    with h5py.File(data_path, "r") as f:
        n_samples = f["Waveforms"].shape[0]
    idx_all = np.arange(n_samples)
    gen = torch.Generator().manual_seed(42)
    _, _, j_test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)
    test_indices = np.sort(np.array(j_test_idx.indices))
    n_test = len(test_indices)
    print(f"Total test waveforms: {n_test}")

    # 4. Batch Processing
    batch_size = 1024
    all_mismatches = []

    with h5py.File(data_path, "r") as f:
        wf_ds = f["Waveforms"]

        for start_idx in range(0, n_test, batch_size):
            end_idx = min(start_idx + batch_size, n_test)
            cur_batch_indices = test_indices[start_idx:end_idx]
            
            # Load waveforms
            w_batch = wf_ds[cur_batch_indices] # shape (B, 2048)
            
            # Ground truth phase
            ph = -np.unwrap(np.angle(w_batch), axis=-1)
            ph_true = ph - ph[:, :1]
            
            # Convert to logfreq
            om = np.diff(ph_true, axis=-1)
            om = np.concatenate([om, om[:, -1:]], axis=-1)
            om = np.maximum(om, omega_floor)
            lf = np.log(om).astype(np.float32)
            
            # Normalize
            lf_norm = (lf - mu) / sd
            if clamp_target > 0:
                lf_norm = np.clip(lf_norm, -clamp_target, clamp_target)
            
            # Forward pass
            with torch.no_grad():
                lf_in = torch.from_numpy(lf_norm).float()
                out = model(lf_in).squeeze(1).numpy()
            
            # Reconstruct phase
            om_pred = np.exp(out.astype(np.float64) * sd + mu)
            ph_pred_cum = np.cumsum(om_pred[:, :-1], axis=-1)
            ph_pred = np.concatenate([np.zeros((len(om_pred), 1)), ph_pred_cum], axis=-1)
            
            # True Amplitude + Predicted Phase
            ampl = np.abs(w_batch).astype(np.float32)
            h_true = torch.polar(torch.from_numpy(ampl), -torch.from_numpy(ph_true.astype(np.float32)))
            h_pred = torch.polar(torch.from_numpy(ampl), -torch.from_numpy(ph_pred.astype(np.float32)))
            
            # Compute Overlaps
            overlaps = TasosOverlapTorch(h_true, h_pred)
            mismatch = torch.clamp(1.0 - overlaps, min=0.0).numpy()
            all_mismatches.append(mismatch)
            
            print(f"Processed [{end_idx}/{n_test}] waveforms... (current batch median: {np.median(mismatch):.2e})")

    mismatches = np.concatenate(all_mismatches)
    
    # Save raw mismatches
    os.makedirs("runs", exist_ok=True)
    out_npy = "runs/monolith_logfreq_mismatches.npy"
    np.save(out_npy, mismatches)
    print(f"\nSaved raw mismatches to {out_npy}")

    # Metrics Summary
    med_m = np.median(mismatches)
    mean_m = np.mean(mismatches)
    p90_m = np.percentile(mismatches, 90)
    p95_m = np.percentile(mismatches, 95)
    p99_m = np.percentile(mismatches, 99)
    max_m = np.max(mismatches)
    
    frac_sub_1e4 = np.mean(mismatches < 1e-4) * 100
    frac_sub_1e3 = np.mean(mismatches < 1e-3) * 100
    frac_sub_1e2 = np.mean(mismatches < 1e-2) * 100

    print("\n" + "=" * 70)
    print("FULL TEST SET MISMATCH STATISTICS (10,240 SIGNALS)")
    print("=" * 70)
    print(f"  Median Mismatch:        {med_m:.4e}  (Overlap: {1 - med_m:.6f})")
    print(f"  Mean Mismatch:          {mean_m:.4e}")
    print(f"  90th Percentile:        {p90_m:.4e}")
    print(f"  95th Percentile:        {p95_m:.4e}")
    print(f"  99th Percentile:        {p99_m:.4e}")
    print(f"  Maximum Mismatch:       {max_m:.4e}")
    print("-" * 70)
    print(f"  Signals < 1e-4:         {frac_sub_1e4:.2f}%")
    print(f"  Signals < 1e-3:         {frac_sub_1e3:.2f}% (Standard surrogate threshold)")
    print(f"  Signals < 1e-2:         {frac_sub_1e2:.2f}%")
    print("=" * 70)

    # Plot Cumulative Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    sorted_m = np.sort(mismatches)
    cdf = np.linspace(0, 1, len(sorted_m))

    ax.semilogx(sorted_m, cdf, color="#1f77b4", linewidth=2.2, label=f"Monolith LogFreq (Median: {med_m:.2e})")
    ax.axvline(1e-3, color="#d9534f", linestyle="--", linewidth=1.5, label="Surrogate Accuracy Goal ($10^{-3}$)")
    ax.axvline(med_m, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Median ({med_m:.2e})")
    ax.axvline(p90_m, color="#ff7f0e", linestyle="-.", linewidth=1.2, label=f"90th Percentile ({p90_m:.2e})")

    ax.set_title("Waveform Mismatch Cumulative Distribution ($1 - \mathcal{M}$)\nMonolith Log-Frequency Autoencoder (Test Set: $N=10,240$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mismatch $1 - \mathcal{M}$ (True Amplitude + Reconstructed Phase)", fontsize=10)
    ax.set_ylabel("Cumulative Fraction", fontsize=10)
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    plt.tight_layout()
    
    out_plot = "runs/monolith_logfreq_mismatch_distribution.png"
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved plot to {out_plot}")

if __name__ == "__main__":
    main()
