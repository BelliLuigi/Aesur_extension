import h5py
import numpy as np
import torch
from models import Autoencoder_CNN_NOBN
from utils import get_phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load trained encoders
ae_ampl = Autoencoder_CNN_NOBN().to(device)
ae_ampl.load_state_dict(torch.load("soap2110_amplitude_best.pt", map_location=device))
ae_ampl.eval()

ae_phase = Autoencoder_CNN_NOBN().to(device)
ae_phase.load_state_dict(torch.load("soap2110_phase_best.pt", map_location=device))
ae_phase.eval()

# 2. Extract latent vectors in batches
with h5py.File("datasets/NRHybSur3dq8_dataset.hdf", "r") as f_in:
    raw_params = f_in["Parameters"][:]  # Shape (102400, 4)
    # Drop column 1 (constant 1.0) -> (102400, 3) with columns: [q, chi1, chi2]
    params = np.delete(raw_params, 1, axis=1).astype(np.float32)
    
    n_samples = len(params)
    z_ampl = np.empty((n_samples, 64), dtype=np.float32)
    z_phase = np.empty((n_samples, 64), dtype=np.float32)
    
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            wave_chunk = f_in["Waveforms"][i:end]
            
            # Amplitude
            a_chunk = torch.from_numpy(np.abs(wave_chunk).astype(np.float32)).unsqueeze(1).to(device)
            z_ampl[i:end] = ae_ampl.encoder(a_chunk).cpu().numpy()
            
            # Phase
            p_chunk = torch.from_numpy(get_phase(wave_chunk).astype(np.float32)).unsqueeze(1).to(device)
            z_phase[i:end] = ae_phase.encoder(p_chunk).cpu().numpy()

# 3. Save offline dataset
with h5py.File("datasets/NRhyblatent.hdf", "w") as f_out:
    f_out.create_dataset("Parameters", data=params)
    f_out.create_dataset("z_amplitude", data=z_ampl)
    f_out.create_dataset("z_phase", data=z_phase)

print("Saved datasets/NRhyblatent.hdf successfully.")
