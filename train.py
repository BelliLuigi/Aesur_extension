import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import separate_phase_ampl
from torch import Tensor
from torch.utils.data import random_split, TensorDataset
import h5py
import numpy as np
from torch.optim import Adam
from utils import train_one_epoch
from torch.utils.tensorboard import SummaryWriter
from models import Autoencoder_CNN_1
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Load dataset from hdf

with h5py.File("nonhyb_noprec_dataset_q8.hdf", "r") as f:
    print("Keys:", list(f.keys()))  # ['Parameters', 'Waveforms']
    waveforms = f["Waveforms"][:100] #numpy.ndarray
    params = f["Parameters"][:100] # same prob

## take wf, split phase and amplitude
splitted_waveforms = separate_phase_ampl(waveforms)

# split in test, train and validation for both
a_train, a_val, a_test = random_split(splitted_waveforms[0], [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))
p_train, p_val, p_test = random_split(splitted_waveforms[1], [0.8,0.1,0.1], generator=torch.Generator().manual_seed(42))

# 1 dataloader for each one
a_train_loader = DataLoader(a_train, batch_size = 128, shuffle= True)
a_val_loader = DataLoader(a_val, batch_size=256, shuffle=False)
a_test_loader = DataLoader(a_test, batch_size=256, shuffle=False)
p_train_loader = DataLoader(p_train, batch_size = 128, shuffle= True)
p_val_loader = DataLoader(p_val, batch_size=256, shuffle=False)
p_test_loader = DataLoader(p_test, batch_size=256, shuffle=False)


model1 = Autoencoder_CNN_1().to(device)


## Loss function
loss_fn = nn.L1Loss(reduction='mean')

# Optimizer function
opt_fn = Adam(model1.parameters(), lr=1e-3, weight_decay=0)

training_loader = a_train_loader
validation_loader = a_val_loader

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/AE_CNN_1_Amplitude_{timestamp}")
EPOCHS = 10
best_vloss = float("inf")

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch + 1}:")
    avg_loss = train_one_epoch(model1, opt_fn, loss_fn, device, epoch, writer, training_loader)

    running_vloss = 0.0
    model1.eval()

    with torch.no_grad():
        for i, vbatch in enumerate(validation_loader):
            vinputs = vbatch.float().to(device)
            if vinputs.ndim == 2:
                vinputs = vinputs.unsqueeze(1)
            voutputs = model1(vinputs)
            vloss = loss_fn(voutputs, vinputs)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print(f"LOSS train: {avg_loss:.6f} | valid: {avg_vloss:.6f}")

    writer.add_scalars(
        "Training and Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch + 1,
    )
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"model_{timestamp}_{epoch}.pt"
        torch.save(model1.state_dict(), model_path)
