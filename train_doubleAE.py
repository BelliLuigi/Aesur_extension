import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_phase
from torch import Tensor
from torch.utils.data import random_split
import os
import h5py
import numpy as np
from torch.optim import Adam, lr_scheduler
from utils import train_one_epoch, L1WeightLoss
from torch.utils.tensorboard import SummaryWriter
from models import Autoencoder_CNN_2
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(device)
    print(f"Allocated GPU: {prop.name} | Compute Capability: {prop.major}.{prop.minor} | VRAM: {prop.total_memory / (1024**3):.1f} GB")
## Load dataset from hdf

filepath = "datasets/NRHybSur3dq8_dataset.hdf" if os.path.exists("datasets/NRHybSur3dq8_dataset.hdf") else "NRHybSur3dq8_dataset.hdf"
condor_filepath = "/lustre/ific.uv.es/ml/uv098/osvaldogramaxo/DANSur_22/data/NRHybSur3dq8_dataset.hdf"
old_filepath = "nonhyb_noprec_dataset_q8.hdf"

with h5py.File(filepath, "r") as f:
    waveforms_ds = f["Waveforms"]
    n_samples, sample_len = waveforms_ds.shape
    #amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
    #phase = np.empty((n_samples, sample_len), dtype=np.float32)
    waveforms = np.empty((n_samples, 2, sample_len), dtype = np.float32  )
    chunk_size = 10000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        waveforms[start:end, 0, :] = np.abs(waveforms_ds[start:end]).astype(np.float32)
        waveforms[start:end, 1, :] = get_phase(waveforms_ds[start:end]).astype(np.float32)
        
wf_tensor = torch.from_numpy(waveforms)


# split in test, train and validation for both
train, val, test = random_split(wf_tensor, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))


# 1 dataloader for each one
train_loader = DataLoader(train, batch_size = 64, shuffle= True)
val_loader = DataLoader(val, batch_size=2048, shuffle=False)
test_loader = DataLoader(test, batch_size=1024, shuffle=False)


model = Autoencoder_CNN_2().to(device)

## Loss function & L2 latent regularization parameter
loss_fn = L1WeightLoss().to(device)
lambda_latent = 1e-5

# Optimizer function
opt_fn = Adam(model.parameters(), lr=7e-3, weight_decay=1e-9)
scheduler = lr_scheduler.ReduceLROnPlateau(opt_fn, factor=0.5, patience=100, threshold=0.0001, threshold_mode='rel', min_lr=1e-6)


def validate(model, val_loader, loss_fn, lambda_latent: float = 0.0):
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vbatch in enumerate(val_loader):
            vinputs = vbatch.float().to(device)
            if vinputs.ndim == 2:
                vinputs = vinputs.unsqueeze(1)
            if lambda_latent > 0.0:
                voutputs, z = model(vinputs, return_latent=True)
                vloss = loss_fn(voutputs, vinputs) + lambda_latent * torch.mean(z ** 2)
            else:
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vinputs)
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/AE_CNN_2_double_{timestamp}")
EPOCHS = 3000
best_vloss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch + 1}:")
    avg_loss = train_one_epoch(model, opt_fn, loss_fn, device, epoch, writer, train_loader, lambda_latent=lambda_latent)
    
    avg_vloss = validate(model, val_loader, loss_fn, lambda_latent=lambda_latent)
    scheduler.step(avg_vloss)
    if epoch % 100 == 99:
        print(f"  DoubleAE | train: {avg_loss:.6f} | valid: {avg_vloss:.6f}")
    current_lr = scheduler.get_last_lr()[0]

    writer.add_scalars(
        "Training and Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss, "lr": current_lr},
        epoch + 1,
    )

    writer.flush()
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), f"model_doubleAE_{timestamp}_best.pt")

