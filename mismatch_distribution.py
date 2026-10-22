#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import Autoencoder_CNN_1, Autoencoder_CNN_NOBN
from utils import get_phase, compute_mismatches, plot_mismatch_distribution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot waveform mismatch distribution for trained Amplitude and Phase Autoencoders."
    )
    parser.add_argument(
        "--ampl_model",
        type=str,
        required=True,
        help="Path to trained amplitude model state_dict (.pt file)",
    )
    parser.add_argument(
        "--phase_model",
        type=str,
        required=True,
        help="Path to trained phase model state_dict (.pt file)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to NRHybSur3dq8 HDF5 dataset. Defaults to datasets/NRHybSur3dq8_dataset.hdf or lustre path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Run title or prefix for output filenames (default: derived from amplitude model name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "train", "all"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Evaluation batch size (default: 1024)",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Whether to save raw mismatches array as .npy file",
    )
    return parser.parse_args()


def load_data(filepath: str, split_choice: str, batch_size: int):
    print(f"Loading dataset from: {filepath}")
    with h5py.File(filepath, "r") as f:
        waveforms_ds = f["Waveforms"]
        n_samples, sample_len = waveforms_ds.shape
        amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
        phase = np.empty((n_samples, sample_len), dtype=np.float32)
        chunk_size = 10000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            amplitude[start:end] = np.abs(waveforms_ds[start:end]).astype(np.float32)
            phase[start:end] = get_phase(waveforms_ds[start:end]).astype(np.float32)

    ampl_tensor = torch.from_numpy(amplitude)
    phase_tensor = torch.from_numpy(phase)
    joint_dataset = TensorDataset(ampl_tensor, phase_tensor)

    if split_choice == "all":
        selected_dataset = joint_dataset
    else:
        j_train, j_val, j_test = random_split(
            joint_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )
        if split_choice == "test":
            selected_dataset = j_test
        elif split_choice == "val":
            selected_dataset = j_val
        else:
            selected_dataset = j_train

    print(f"Evaluating split '{split_choice}' ({len(selected_dataset)} waveforms)")
    loader = DataLoader(selected_dataset, batch_size=batch_size, shuffle=False)
    return loader


def main():
    args = parse_args()

    # Determine title / output prefix
    title = args.title
    if not title:
        base = os.path.basename(args.ampl_model)
        for suffix in ["_amplitude_best.pt", "_amplitude_checkpoint.pt", "_best.pt", ".pt"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        title = base or "eval_run"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset path resolution
    dataset_path = args.dataset
    if not dataset_path:
        default_local = "datasets/NRHybSur3dq8_dataset.hdf"
        default_lustre = "/lustre/ific.uv.es/ml/uv098/osvaldogramaxo/DANSur_22/data/NRHybSur3dq8_dataset.hdf"
        dataset_path = default_local if os.path.exists(default_local) else default_lustre

    loader = load_data(dataset_path, args.split, args.batch_size)

    # Initialize models and load weights
    print(f"Loading Amplitude model: {args.ampl_model}")
    model_amplitude = Autoencoder_CNN_NOBN().to(device) #Autoencoder_CNN_1().to(device)
    state_a = torch.load(args.ampl_model, map_location=device)
    if "model_state_dict" in state_a:
        state_a = state_a["model_state_dict"]
    model_amplitude.load_state_dict(state_a)

    print(f"Loading Phase model:     {args.phase_model}")
    model_phase = Autoencoder_CNN_NOBN().to(device) #Autoencoder_CNN_1().to(device)
    state_p = torch.load(args.phase_model, map_location=device)
    if "model_state_dict" in state_p:
        state_p = state_p["model_state_dict"]
    model_phase.load_state_dict(state_p)

    # Compute mismatches
    print("Computing waveform mismatches (1 - Overlap)...")
    mismatches = compute_mismatches(model_amplitude, model_phase, loader, device=device)

    # Statistics
    med = float(np.median(mismatches))
    mean_val = float(np.mean(mismatches))
    max_val = float(np.max(mismatches))
    p90 = float(np.percentile(mismatches, 90))
    p99 = float(np.percentile(mismatches, 99))

    print(f"\n=== Mismatch Evaluation Summary ({len(mismatches)} waveforms) ===")
    print(f"  Median Mismatch:     {med:.3e}")
    print(f"  Mean Mismatch:       {mean_val:.3e}")
    print(f"  Max Mismatch:        {max_val:.3e}")
    print(f"  90th Percentile:     {p90:.3e}")
    print(f"  99th Percentile:     {p99:.3e}")

    # Plot distribution
    output_plot = f"{title}_mismatch_distribution.png"
    plot_mismatch_distribution(mismatches, run_title=title, output_path=output_plot)

    if args.save_npy:
        npy_path = f"{title}_mismatches.npy"
        np.save(npy_path, mismatches)
        print(f"Raw mismatches saved: {npy_path}")


if __name__ == "__main__":
    main()
