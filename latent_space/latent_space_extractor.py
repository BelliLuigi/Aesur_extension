import sys
from pathlib import Path

# Add project root to sys.path so models and utils can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import h5py
import numpy as np
from models import Autoencoder_CNN_1
from utils import get_phase

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(device)
    print(f"Allocated GPU: {prop.name} | Compute Capability: {prop.major}.{prop.minor} | VRAM: {prop.total_memory / (1024**3):.1f} GB")

# weight location paths
model_a_weight = PROJECT_ROOT / 'model_weights/model1_ampl_20261015_113003_best.pt'
model_p_weight = PROJECT_ROOT / 'model_weights/model2_phase_20261015_113003_best.pt'

## dataset path
condor_filepath = "/lustre/ific.uv.es/ml/uv098/osvaldogramaxo/DANSur_22/data/NRHybSur3dq8_dataset.hdf"
ugly_filepath = "nonhyb_noprec_dataset_q8.hdf"
filepath = PROJECT_ROOT / 'datasets/NRHybSur3dq8_dataset.hdf'


# load models in device 
model_a = Autoencoder_CNN_1(negative_slope=0.1619).to(device)
model_a.load_state_dict(torch.load(model_a_weight, map_location = device))
model_p = Autoencoder_CNN_1(negative_slope=0.0859).to(device)
model_p.load_state_dict(torch.load(model_p_weight, map_location=device))

# no gradients
model_a.eval()
model_p.eval()

#load test set

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

# split in test, train and validation for both
_,_, a_test = random_split(ampl_tensor, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))
_,_, p_test = random_split(phase_tensor, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))

# dataloaders
a_test_loader = DataLoader(a_test, batch_size=256, shuffle=False)
p_test_loader = DataLoader(p_test, batch_size=256, shuffle=False)


latent_amplitude_vectors = []
latent_phase_vectors = []

with torch.no_grad():
    for batch in a_test_loader:
        x = batch.float().unsqueeze(1).to(device)
        latent_amplitude_vectors.append(model_a.encoder(x))

    Z_ampl = torch.cat(latent_amplitude_vectors, dim=0)

    for batch in p_test_loader:
        x = batch.float().unsqueeze(1).to(device)
        latent_phase_vectors.append(model_p.encoder(x))

    Z_phase = torch.cat(latent_phase_vectors, dim=0)

out_dir = Path(__file__).resolve().parent
np.save(out_dir / "Z_ampl.npy", Z_ampl.cpu().numpy())
np.save(out_dir / "Z_phase.npy", Z_phase.cpu().numpy())








