import sys
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models import Autoencoder_CNN_1
from utils import get_phase

def main():
    print("=== ESTENSIONE E VALUTAZIONE MODELLI SINGOLI GELU ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    model_a_weight = PROJECT_ROOT / 'model_weights/model1_ampl_gelu.pt'
    model_p_weight = PROJECT_ROOT / 'model_weights/model2_phase_gelu.pt'
    filepath = PROJECT_ROOT / 'datasets/NRHybSur3dq8_dataset.hdf'
    out_dir = PROJECT_ROOT / 'model_analysis'

    # Load models
    print(f"Caricamento pesi ampiezza: {model_a_weight.name}")
    model_a = Autoencoder_CNN_1().to(device)
    model_a.load_state_dict(torch.load(model_a_weight, map_location=device))
    model_a.eval()

    print(f"Caricamento pesi fase: {model_p_weight.name}")
    model_p = Autoencoder_CNN_1().to(device)
    model_p.load_state_dict(torch.load(model_p_weight, map_location=device))
    model_p.eval()

    # Load test set
    print(f"Caricamento dataset: {filepath.name}")
    t0 = time.time()
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
    print(f"Dataset caricato in {time.time()-t0:.2f}s.")

    # Split: 80% train, 10% val, 10% test (seed 42)
    generator = torch.Generator().manual_seed(42)
    _, _, a_test = random_split(ampl_tensor, [0.8, 0.1, 0.1], generator=generator)
    generator = torch.Generator().manual_seed(42)
    _, _, p_test = random_split(phase_tensor, [0.8, 0.1, 0.1], generator=generator)

    a_test_loader = DataLoader(a_test, batch_size=256, shuffle=False)
    p_test_loader = DataLoader(p_test, batch_size=256, shuffle=False)

    print(f"Test set: {len(a_test)} campioni.")

    # 1. Amplitude Evaluation
    print("Inference e calcolo metriche Ampiezza...")
    t0 = time.time()
    latent_a = []
    recon_a = []
    orig_a = []

    with torch.no_grad():
        for batch in a_test_loader:
            orig_a.append(batch.numpy())
            x = batch.float().unsqueeze(1).to(device)
            out, z = model_a(x, return_latent=True)
            latent_a.append(z.cpu().numpy())
            recon_a.append(out.squeeze(1).cpu().numpy())

    Z_ampl = np.concatenate(latent_a, axis=0)
    A_orig = np.concatenate(orig_a, axis=0)
    A_rec = np.concatenate(recon_a, axis=0)
    print(f"Ampiezza completata in {time.time()-t0:.2f}s.")

    # Metrics Amplitude
    mae_ampl = np.mean(np.abs(A_orig - A_rec), axis=1)
    mse_ampl = np.mean((A_orig - A_rec) ** 2, axis=1)
    rel_l1_ampl = np.sum(np.abs(A_orig - A_rec), axis=1) / (np.sum(np.abs(A_orig), axis=1) + 1e-12)
    temporal_mae_ampl = np.mean(np.abs(A_orig - A_rec), axis=0)

    # 2. Phase Evaluation
    print("Inference e calcolo metriche Fase...")
    t0 = time.time()
    latent_p = []
    recon_p = []
    orig_p = []

    with torch.no_grad():
        for batch in p_test_loader:
            orig_p.append(batch.numpy())
            x = batch.float().unsqueeze(1).to(device)
            out, z = model_p(x, return_latent=True)
            latent_p.append(z.cpu().numpy())
            recon_p.append(out.squeeze(1).cpu().numpy())

    Z_phase = np.concatenate(latent_p, axis=0)
    P_orig = np.concatenate(orig_p, axis=0)
    P_rec = np.concatenate(recon_p, axis=0)
    print(f"Fase completata in {time.time()-t0:.2f}s.")

    # Metrics Phase
    mae_phase = np.mean(np.abs(P_orig - P_rec), axis=1)
    mse_phase = np.mean((P_orig - P_rec) ** 2, axis=1)
    rel_l1_phase = np.sum(np.abs(P_orig - P_rec), axis=1) / (np.sum(np.abs(P_orig), axis=1) + 1e-12)
    temporal_mae_phase = np.mean(np.abs(P_orig - P_rec), axis=0)

    # Save Arrays
    print("Salvataggio array numpy in model_analysis/...")
    np.save(out_dir / "Z_ampl_gelu.npy", Z_ampl)
    np.save(out_dir / "Z_phase_gelu.npy", Z_phase)
    np.save(out_dir / "mae_ampl_gelu.npy", mae_ampl)
    np.save(out_dir / "mse_ampl_gelu.npy", mse_ampl)
    np.save(out_dir / "rel_l1_ampl_gelu.npy", rel_l1_ampl)
    np.save(out_dir / "temporal_mae_ampl_gelu.npy", temporal_mae_ampl)

    np.save(out_dir / "mae_phase_gelu.npy", mae_phase)
    np.save(out_dir / "mse_phase_gelu.npy", mse_phase)
    np.save(out_dir / "rel_l1_phase_gelu.npy", rel_l1_phase)
    np.save(out_dir / "temporal_mae_phase_gelu.npy", temporal_mae_phase)

    # Save sample wave indices (best, median, 90th percentile, worst)
    sample_indices = {
        "best_ampl_idx": int(np.argmin(mae_ampl)),
        "median_ampl_idx": int(np.argsort(mae_ampl)[len(mae_ampl) // 2]),
        "p90_ampl_idx": int(np.argsort(mae_ampl)[int(len(mae_ampl) * 0.90)]),
        "worst_ampl_idx": int(np.argmax(mae_ampl)),
        "best_phase_idx": int(np.argmin(mae_phase)),
        "median_phase_idx": int(np.argsort(mae_phase)[len(mae_phase) // 2]),
        "p90_phase_idx": int(np.argsort(mae_phase)[int(len(mae_phase) * 0.90)]),
        "worst_phase_idx": int(np.argmax(mae_phase)),
    }
    np.save(out_dir / "sample_indices_gelu.npy", sample_indices)

    # Summary Output
    print("\n" + "=" * 50)
    print("RIASSUNTO PRESTAZIONI GELU SINGLE AUTOENCODERS")
    print("=" * 50)
    print(f"AMPIEZZA (10,240 test samples):")
    print(f"  MAE medio:       {np.mean(mae_ampl):.6f}")
    print(f"  MAE mediano:     {np.median(mae_ampl):.6f}")
    print(f"  MAE std:         {np.std(mae_ampl):.6f}")
    print(f"  MSE medio:       {np.mean(mse_ampl):.6e}")
    print(f"  Rel L1 medio:    {np.mean(rel_l1_ampl)*100:.3f}%")
    print(f"  Rel L1 mediano:  {np.median(rel_l1_ampl)*100:.3f}%")

    print(f"\nFASE (10,240 test samples):")
    print(f"  MAE medio:       {np.mean(mae_phase):.6f}")
    print(f"  MAE mediano:     {np.median(mae_phase):.6f}")
    print(f"  MAE std:         {np.std(mae_phase):.6f}")
    print(f"  MSE medio:       {np.mean(mse_phase):.6e}")
    print(f"  Rel L1 medio:    {np.mean(rel_l1_phase)*100:.3f}%")
    print(f"  Rel L1 mediano:  {np.median(rel_l1_phase)*100:.3f}%")

    # Latent space stats
    var_a = np.var(Z_ampl, axis=0)
    var_p = np.var(Z_phase, axis=0)
    print(f"\nSPAZIO LATENTE AMPIEZZA:")
    print(f"  Dimensioni con var > 1e-4: {np.sum(var_a > 1e-4)} / 64")
    print(f"  Dimensioni con var > 1e-3: {np.sum(var_a > 1e-3)} / 64")
    print(f"  Dimensioni con var > 0.01: {np.sum(var_a > 0.01)} / 64")
    print(f"  Varianza max: {np.max(var_a):.4f}, min: {np.min(var_a):.6e}")

    print(f"\nSPAZIO LATENTE FASE:")
    print(f"  Dimensioni con var > 1e-4: {np.sum(var_p > 1e-4)} / 64")
    print(f"  Dimensioni con var > 1e-3: {np.sum(var_p > 1e-3)} / 64")
    print(f"  Dimensioni con var > 0.01: {np.sum(var_p > 0.01)} / 64")
    print(f"  Varianza max: {np.max(var_p):.4f}, min: {np.min(var_p):.6e}")
    print("=" * 50)

if __name__ == "__main__":
    main()
