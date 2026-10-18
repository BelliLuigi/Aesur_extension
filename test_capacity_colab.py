import torch
import torch.nn as nn
import numpy as np
import h5py
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def get_phase(dataset):
    phase_non = -1 * np.unwrap(np.angle(dataset)) / 100
    out = phase_non
    phase = out - out[..., np.newaxis, 0]
    return phase

# --- Model Definitions ---
class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(32, 16, kernel_size=40, stride=4, padding=18, bias=False)
        self.bn2 = nn.BatchNorm1d(16)
        self.c3 = nn.Conv1d(16, 8, kernel_size=40, stride=4, padding=18, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc = nn.Linear(512, 64, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        return self.fc(torch.flatten(x, start_dim=1))

class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 256, bias=False)
        self.bn0 = nn.BatchNorm1d(256)
        self.ct1 = nn.ConvTranspose1d(1, 32, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.ct2 = nn.ConvTranspose1d(32, 32, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.ct3 = nn.ConvTranspose1d(32, 8, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.ct4 = nn.Conv1d(8, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn0(self.fc(x))).unsqueeze(1)
        x = self.act(self.bn1(self.ct1(x)))
        x = self.act(self.bn2(self.ct2(x)))
        x = self.act(self.bn3(self.ct3(x)))
        return self.ct4(x)

class Autoencoder_CNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder1()
        self.decoder = Decoder1()
    def forward(self, x):
        return self.decoder(self.encoder(x))

class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(2, 64, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 32, kernel_size=40, stride=4, padding=18, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.c3 = nn.Conv1d(32, 16, kernel_size=40, stride=4, padding=18, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc = nn.Linear(1024, 64, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        return self.fc(torch.flatten(x, start_dim=1))

class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 512, bias=False)
        self.bn0 = nn.BatchNorm1d(512)
        self.ct1 = nn.ConvTranspose1d(2, 32, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.ct2 = nn.ConvTranspose1d(32, 32, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.ct3 = nn.ConvTranspose1d(32, 8, kernel_size=40, stride=2, padding=19, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.ct4 = nn.Conv1d(8, 2, kernel_size=7, stride=1, padding=3, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn0(self.fc(x))).view(-1, 2, 256)
        x = self.act(self.bn1(self.ct1(x)))
        x = self.act(self.bn2(self.ct2(x)))
        x = self.act(self.bn3(self.ct3(x)))
        return self.ct4(x)

class Autoencoder_CNN_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2()
    def forward(self, x):
        return self.decoder(self.encoder(x))

class DenseAE(nn.Module):
    def __init__(self, in_features=2048, latent_dim=64, units=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features, 8 * units), nn.LeakyReLU(0.3),
            nn.Linear(8 * units, 4 * units), nn.LeakyReLU(0.3),
            nn.Linear(4 * units, 2 * units), nn.LeakyReLU(0.3),
            nn.Linear(2 * units, units), nn.LeakyReLU(0.3),
            nn.Linear(units, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, units), nn.LeakyReLU(0.3),
            nn.Linear(units, 2 * units), nn.LeakyReLU(0.3),
            nn.Linear(2 * units, 4 * units), nn.LeakyReLU(0.3),
            nn.Linear(4 * units, 8 * units), nn.LeakyReLU(0.3),
            nn.Linear(8 * units, in_features)
        )
    def forward(self, x):
        return self.dec(self.enc(x))

# --- Load Dataset ---
dataset_path = "/content/drive/MyDrive/NRhybSur3dq8_dataset.hdf"
print(f"Loading waveforms from {dataset_path}...")
with h5py.File(dataset_path, "r") as f:
    raw = f["Waveforms"][:200]

n_samples, sample_len = raw.shape
print(f"Loaded {n_samples} waveforms of length {sample_len}")

ampl_np = np.abs(raw).astype(np.float32)
phase_np = get_phase(raw).astype(np.float32)

double_np = np.zeros((n_samples, 2, sample_len), dtype=np.float32)
double_np[:, 0, :] = ampl_np
double_np[:, 1, :] = phase_np

# Take batch of 64 waveforms for overfitting capacity diagnostic
B = 64
b_double = torch.from_numpy(double_np[:B]).to(device)
b_ampl = torch.from_numpy(ampl_np[:B]).unsqueeze(1).to(device)
b_phase = torch.from_numpy(phase_np[:B]).unsqueeze(1).to(device)
b_ampl_flat = torch.from_numpy(ampl_np[:B]).to(device)

loss_fn = nn.L1Loss()

print("\n" + "="*50)
print(f"TEST 1: SINGLE-BATCH OVERFITTING TEST (N={B} waveforms, 500 steps)")
print("="*50)

def train_overfit(model, x, steps=500, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        opt.step()
        if step in [1, 50, 100, 200, 350, 500]:
            losses.append((step, loss.item()))
    return losses

print("\n1. Autoencoder_CNN_2 (Double AE, ~261k params):")
m2 = Autoencoder_CNN_2().to(device)
for step, l in train_overfit(m2, b_double):
    print(f"   Step {step:3d} | L1 Loss: {l:.6e}")

print("\n2. Autoencoder_CNN_1 (Amplitude only, ~129k params):")
m1_a = Autoencoder_CNN_1().to(device)
for step, l in train_overfit(m1_a, b_ampl):
    print(f"   Step {step:3d} | L1 Loss: {l:.6e}")

print("\n3. Autoencoder_CNN_1 (Phase only, ~129k params):")
m1_p = Autoencoder_CNN_1().to(device)
for step, l in train_overfit(m1_p, b_phase):
    print(f"   Step {step:3d} | L1 Loss: {l:.6e}")

print("\n4. DenseAE (Paper architecture, ~3.0M params on Amplitude):")
m_dense = DenseAE().to(device)
for step, l in train_overfit(m_dense, b_ampl_flat):
    print(f"   Step {step:3d} | L1 Loss: {l:.6e}")

print("\n" + "="*50)
print("TEST 2: LINEAR PCA BASELINE (64 components on 64 waveforms)")
print("="*50)
U, S, Vh = torch.linalg.svd(b_ampl_flat, full_matrices=False)
k = 64
recon_pca = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
pca_l1 = torch.mean(torch.abs(recon_pca - b_ampl_flat)).item()
print(f"PCA (k=64) Reconstruction L1 Loss: {pca_l1:.6e}")
