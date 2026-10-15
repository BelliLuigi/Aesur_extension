import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_phase
from torch import Tensor
from torch.utils.data import random_split
import h5py
import numpy as np
from torch.optim import Adam, lr_scheduler
from utils import train_one_epoch
from torch.utils.tensorboard import SummaryWriter
from models import Autoencoder_CNN_1
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Load dataset from hdf

filepath = "/lustre/ific.uv.es/ml/uv098/osvaldogramaxo/DANSur_22/data/NRHybSur3dq8_dataset.hdf"
old_filepath = "nonhyb_noprec_dataset_q8.hdf"

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
a_train, a_val, a_test = random_split(ampl_tensor, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))
p_train, p_val, p_test = random_split(phase_tensor, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))

# 1 dataloader for each one
a_train_loader = DataLoader(a_train, batch_size = 128, shuffle= True)
a_val_loader = DataLoader(a_val, batch_size=256, shuffle=False)
a_test_loader = DataLoader(a_test, batch_size=256, shuffle=False)
p_train_loader = DataLoader(p_train, batch_size = 64, shuffle= True)
p_val_loader = DataLoader(p_val, batch_size=256, shuffle=False)
p_test_loader = DataLoader(p_test, batch_size=256, shuffle=False)


model1 = Autoencoder_CNN_1(negative_slope=0.1619).to(device) # amplitude
model2 = Autoencoder_CNN_1(negative_slope=0.0859).to(device) #phase

## Loss function
loss_fn1 = nn.L1Loss(reduction='mean')
loss_fn2 = nn.L1Loss(reduction='mean')

# Optimizer function
opt_fn1 = Adam(model1.parameters(), lr=0.0068, weight_decay=1.4e-6)
scheduler1 = lr_scheduler.ReduceLROnPlateau(opt_fn1, factor=0.2, patience=5, threshold=0.01, threshold_mode='rel')
opt_fn2 = Adam(model2.parameters(), lr=0.0071, weight_decay=0.0098)
scheduler2 = lr_scheduler.ReduceLROnPlateau(opt_fn2, factor=0.2, patience=6, threshold=0.01, threshold_mode='rel')


def validate(model, val_loader, loss_fn):
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vbatch in enumerate(val_loader):
            vinputs = vbatch.float().to(device)
            if vinputs.ndim == 2:
                vinputs = vinputs.unsqueeze(1)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vinputs)
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer1 = SummaryWriter(f"runs/AE_CNN_1_Amplitude_{timestamp}")
writer2 = SummaryWriter(f"runs/AE_CNN_1_Phase_{timestamp}")
EPOCHS = 3000
best_vloss1 = float("inf")
best_vloss2 = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch + 1}:")
    print("  [Amplitude]")
    avg_loss1 = train_one_epoch(model1, opt_fn1, loss_fn1, device, epoch, writer1, a_train_loader)
    print("  [Phase]")
    avg_loss2 = train_one_epoch(model2, opt_fn2, loss_fn2, device, epoch, writer2, p_train_loader)

    avg_vloss1 = validate(model1, a_val_loader, loss_fn1)
    scheduler1.step(avg_vloss1)
    avg_vloss2 = validate(model2, p_val_loader, loss_fn2)
    scheduler2.step(avg_vloss2)
    if epoch % 100 == 99:
        print(f"  Amplitude | train: {avg_loss1:.6f} | valid: {avg_vloss1:.6f}")
        print(f"  Phase     | train: {avg_loss2:.6f} | valid: {avg_vloss2:.6f}")
    current_lr1 = scheduler1.get_last_lr()[0]
    current_lr2 = scheduler2.get_last_lr()[0]

    writer1.add_scalars(
        "Training and Validation Loss",
        {"Training": avg_loss1, "Validation": avg_vloss1, "lr": current_lr1},
        epoch + 1,
    )
    writer1.flush()

    writer2.add_scalars(
        "Training and Validation Loss",
        {"Training": avg_loss2, "Validation": avg_vloss2, "lr": current_lr2},
        epoch + 1,
    )
    writer2.flush()

    if avg_vloss1 < best_vloss1:
        best_vloss1 = avg_vloss1
        torch.save(model1.state_dict(), f"model1_ampl_{timestamp}_best.pt")

    if avg_vloss2 < best_vloss2:
        best_vloss2 = avg_vloss2
        torch.save(model2.state_dict(), f"model2_phase_{timestamp}_best.pt")

