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
import pandas as pd
import pytorch_optimizer as opt_add
from siren import Sine
from siren.init import siren_uniform_
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


# ==============================================================================
# SECTION 1: Signal Processing & Physics Utilities
# ==============================================================================

def get_phase(dataset: np.ndarray) -> np.ndarray:
    """
    Computes unwrapped and zero-aligned phase scaled by 1/100 from complex waveforms.
    """
    phase_non = -1 * np.unwrap(np.angle(dataset)) / 100
    phase = phase_non - phase_non[..., np.newaxis, 0]
    return phase


def TasosOverlapTorch(
    h1: torch.Tensor,
    h2: torch.Tensor,
    dt: float = 2 * 4.925794970773135e-06,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes normalized frequency-domain overlap in [-1, 1] between complex waveforms.
    Promotes FFT and inner products to complex128/float64 to preserve precision down to 1e-12.
    """
    h1_d = h1.to(torch.complex128)
    h2_d = h2.to(torch.complex128)
    h1_f = torch.fft.fft(h1_d, dim=-1) * dt
    h2_f = torch.fft.fft(h2_d, dim=-1) * dt
    df = 1.0 / (h1.shape[-1] * dt)
    sig_norm = 4.0 * df
    sig1 = torch.sqrt(torch.sum((h1_f.conj() * h1_f).real, dim=-1) * sig_norm + eps)
    sig2 = torch.sqrt(torch.sum((h2_f.conj() * h2_f).real, dim=-1) * sig_norm + eps)
    norm = 1.0 / (sig1 * sig2 + eps)
    inner = torch.sum((h1_f.conj() * h2_f).real, dim=-1)
    overl = 4.0 * df * inner * norm
    return torch.clamp(overl, min=-1.0, max=1.0)


def compute_mismatches(
    model_amplitude: nn.Module,
    model_phase: nn.Module,
    loader: DataLoader,
    device: torch.device = None,
    norm: float = 100.0,
    norm_stats: dict = None,
) -> np.ndarray:
    """
    Evaluates Amplitude and Phase Autoencoders over a DataLoader yielding (amplitude, phase),
    reconstructs polar waveforms h = A * exp(-i * phase * norm), and returns mismatches (1 - Overlap).
    """
    if device is None:
        device = next(model_amplitude.parameters()).device
    model_amplitude.eval()
    model_phase.eval()
    dtype = next(model_amplitude.parameters()).dtype

    if norm_stats is not None:
        mu_a = torch.as_tensor(norm_stats["mu_ampl"], device=device, dtype=dtype).view(1, -1)
        sd_a = torch.as_tensor(norm_stats["sd_ampl"], device=device, dtype=dtype).view(1, -1)
        mu_p = torch.as_tensor(norm_stats["mu_phase"], device=device, dtype=dtype).view(1, -1)
        sd_p = torch.as_tensor(norm_stats["sd_phase"], device=device, dtype=dtype).view(1, -1)
    else:
        mu_a = sd_a = mu_p = sd_p = None

    mismatches = []
    with torch.no_grad():
        for b_ampl, b_phase in loader:
            b_ampl = b_ampl.to(device=device, dtype=dtype)
            b_phase = b_phase.to(device=device, dtype=dtype)
            if b_ampl.ndim == 2:
                b_ampl = b_ampl.unsqueeze(1)
            if b_phase.ndim == 2:
                b_phase = b_phase.unsqueeze(1)

            pred_ampl = model_amplitude(b_ampl)
            pred_phase = model_phase(b_phase)

            a_true = b_ampl.squeeze(1)
            p_true = b_phase.squeeze(1)
            a_pred = pred_ampl.squeeze(1)
            p_pred = pred_phase.squeeze(1)

            if norm_stats is not None:
                a_true = a_true * sd_a + mu_a
                a_pred = a_pred * sd_a + mu_a
                p_true = p_true * sd_p + mu_p
                p_pred = p_pred * sd_p + mu_p

            true_wf = torch.polar(a_true, -p_true * norm)
            pred_wf = torch.polar(torch.clamp(a_pred, min=0.0), -p_pred * norm)

            match = TasosOverlapTorch(true_wf, pred_wf)
            mismatch = torch.clamp(1.0 - match, min=0.0)
            mismatches.extend(mismatch.cpu().numpy().tolist())

    return np.array(mismatches, dtype=np.float64)


# ==============================================================================
# SECTION 2: Model Architecture & SIREN Initialization
# ==============================================================================

class EncoderMono(nn.Module):
    """
    1D CNN Encoder with widened bottleneck (latent shape = (14, 18), dimension = 252)
    without BatchNorm (bias=True).
    Dimensions: 2048 -> 1043 -> 530 -> 270 -> 138 -> 35 -> 18.
    Channels: 1 -> 32 -> 32 -> 64 -> 128 -> 32 -> 14.
    Parameters: 130,350
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, kernel_size=40, stride=2, padding=38, padding_mode="replicate")
        self.c2 = nn.Conv1d(32, 32, kernel_size=20, stride=2, padding=18, padding_mode="replicate")
        self.c3 = nn.Conv1d(32, 64, kernel_size=12, stride=2, padding=10, padding_mode="replicate")
        self.c4 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=6, padding_mode="replicate")
        self.c5 = nn.Conv1d(128, 32, kernel_size=4, stride=4, padding=2, padding_mode="replicate")
        self.c6 = nn.Conv1d(32, 14, kernel_size=4, stride=2, padding=2, padding_mode="replicate")
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.act(self.c5(x))
        return self.c6(x)


class ResBlock1D(nn.Module):
    """
    Intra-layer 1D residual block (x + F(x)) with GELU activation for the Amplitude decoder.
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same", padding_mode="replicate"
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same", padding_mode="replicate"
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DecoderMono(nn.Module):
    """
    1D CNN Decoder with GELU activations for Amplitude reconstruction.
    Dimensions: 18 -> 35 -> 138 -> 270 -> 530 -> 1043 -> 2048.
    Channels: 14 -> 32 -> 128 -> 64 -> 48 -> 32 -> 32 -> 1.
    Parameters: 271,121
    """

    def __init__(self):
        super().__init__()
        self.u6 = nn.Upsample(size=35, mode="linear", align_corners=False)
        self.c6 = nn.Conv1d(14, 32, kernel_size=5, padding="same", padding_mode="replicate")
        self.u5 = nn.Upsample(size=138, mode="linear", align_corners=False)
        self.c5 = nn.Conv1d(32, 128, kernel_size=7, padding="same", padding_mode="replicate")
        self.u4 = nn.Upsample(size=270, mode="linear", align_corners=False)
        self.c4 = nn.Conv1d(128, 64, kernel_size=9, padding="same", padding_mode="replicate")
        self.u3 = nn.Upsample(size=530, mode="linear", align_corners=False)
        self.c3 = nn.Conv1d(64, 48, kernel_size=13, padding="same", padding_mode="replicate")
        self.res3 = ResBlock1D(channels=48, kernel_size=5)
        self.u2 = nn.Upsample(size=1043, mode="linear", align_corners=False)
        self.c2 = nn.Conv1d(48, 32, kernel_size=21, padding="same", padding_mode="replicate")
        self.res2 = ResBlock1D(channels=32, kernel_size=7)
        self.u1 = nn.Upsample(size=2048, mode="linear", align_corners=False)
        self.c1 = nn.Conv1d(32, 32, kernel_size=41, padding="same", padding_mode="replicate")
        self.res1 = ResBlock1D(channels=32, kernel_size=7)
        self.head = nn.Conv1d(32, 1, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(-1, 14, 18)
        x = self.act(self.c6(self.u6(x)))
        x = self.act(self.c5(self.u5(x)))
        x = self.act(self.c4(self.u4(x)))
        x = self.res3(self.act(self.c3(self.u3(x))))
        x = self.res2(self.act(self.c2(self.u2(x))))
        x = self.res1(self.act(self.c1(self.u1(x))))
        return self.head(x)


class SirenResBlock1D(nn.Module):
    """
    Intra-layer 1D residual block (x + conv2(Sine(w0)(conv1(x)))) at fixed temporal resolution.
    conv1 is followed by Sine(w0); conv2 is a linear projection into the residual stream.
    """

    def __init__(self, channels: int, kernel_size: int = 7, w0: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same", padding_mode="replicate"
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same", padding_mode="replicate"
        )
        self.act = Sine(w0=w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DecoderSirenMono(nn.Module):
    """
    1D CNN Decoder using siren-torch Sine activations and principled SIREN initialization
    for Phase reconstruction.
    Dimensions: 18 -> 35 -> 138 -> 270 -> 530 -> 1043 -> 2048.
    Channels: 14 -> 32 -> 128 -> 64 -> 48 -> 32 -> 32 -> 1.
    Parameters: 271,121

    SIREN activation & initialization rules applied:
      - First layer (c6 -> Sine(w0_initial)): receives encoder latents (Var ~ 1.0, ~2x arcsine var),
        so weights use siren_uniform_(c=3.0 / w0_initial^2) to keep Var(w0_initial * pre) ~ 1.0.
      - Hidden layers (c5..c1, res*.conv1 -> Sine(w0)): receive Sine outputs (arcsine Var ~ 0.5),
        so weights use siren_uniform_(c=6.0 / w0^2) per Sitzmann et al. (2020) Theorem 1.8.
      - Residual output convs (res*.conv2): linear projections added to the residual stream (not
        followed by Sine), initialized with siren_uniform_(c=6.0) * 0.1 to preserve variance.
      - Final head (self.head): unactivated linear projection for unbounded standardized targets,
        initialized with siren_uniform_(c=6.0) so Var(head(h)) ~ 1.0.
    """

    def __init__(self, w0_initial: float = 1.0, w0: float = 1.0):
        super().__init__()
        if w0_initial <= 0.0 or w0 <= 0.0:
            raise ValueError(f"w0_initial and w0 must be positive, got {w0_initial=}, {w0=}.")
        self.w0_initial = float(w0_initial)
        self.w0 = float(w0)

        self.u6 = nn.Upsample(size=35, mode="linear", align_corners=False)
        self.c6 = nn.Conv1d(14, 32, kernel_size=5, padding="same", padding_mode="replicate")
        self.act_init = Sine(w0=self.w0_initial)

        self.u5 = nn.Upsample(size=138, mode="linear", align_corners=False)
        self.c5 = nn.Conv1d(32, 128, kernel_size=7, padding="same", padding_mode="replicate")

        self.u4 = nn.Upsample(size=270, mode="linear", align_corners=False)
        self.c4 = nn.Conv1d(128, 64, kernel_size=9, padding="same", padding_mode="replicate")

        self.u3 = nn.Upsample(size=530, mode="linear", align_corners=False)
        self.c3 = nn.Conv1d(64, 48, kernel_size=13, padding="same", padding_mode="replicate")
        self.res3 = SirenResBlock1D(channels=48, kernel_size=5, w0=self.w0)

        self.u2 = nn.Upsample(size=1043, mode="linear", align_corners=False)
        self.c2 = nn.Conv1d(48, 32, kernel_size=21, padding="same", padding_mode="replicate")
        self.res2 = SirenResBlock1D(channels=32, kernel_size=7, w0=self.w0)

        self.u1 = nn.Upsample(size=2048, mode="linear", align_corners=False)
        self.c1 = nn.Conv1d(32, 32, kernel_size=41, padding="same", padding_mode="replicate")
        self.res1 = SirenResBlock1D(channels=32, kernel_size=7, w0=self.w0)

        self.head = nn.Conv1d(32, 1, kernel_size=1)
        self.act = Sine(w0=self.w0)

        init_decoder_siren(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(-1, 14, 18)
        x = self.act_init(self.c6(self.u6(x)))
        x = self.act(self.c5(self.u5(x)))
        x = self.act(self.c4(self.u4(x)))
        x = self.res3(self.act(self.c3(self.u3(x))))
        x = self.res2(self.act(self.c2(self.u2(x))))
        x = self.res1(self.act(self.c1(self.u1(x))))
        return self.head(x)


def init_kaiming_weights(module: nn.Module) -> None:
    """
    Initializes GELU Conv1d layers with Kaiming normal (fan_in, nonlinearity='relu'),
    zeroes biases, and scales ResBlock1D.conv2 weights by 0.1x.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    for m in module.modules():
        if isinstance(m, ResBlock1D):
            with torch.no_grad():
                m.conv2.weight.mul_(0.1)


def init_decoder_siren(decoder: DecoderSirenMono) -> None:
    """
    Initializes DecoderSirenMono using siren_uniform_ matched to each layer's role and w0.
    """
    # 1. First decoder layer (latent -> Sine(w0_initial))
    siren_uniform_(decoder.c6.weight, mode="fan_in", c=3.0 / (decoder.w0_initial ** 2))
    if decoder.c6.bias is not None:
        nn.init.zeros_(decoder.c6.bias)

    # 2. Hidden upsampling convs (Sine -> Sine(w0))
    c_hidden = 6.0 / (decoder.w0 ** 2)
    for conv in (decoder.c5, decoder.c4, decoder.c3, decoder.c2, decoder.c1):
        siren_uniform_(conv.weight, mode="fan_in", c=c_hidden)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    # 3. Residual blocks: conv1 feeds Sine(w0); conv2 projects linearly back to residual stream
    for res in (decoder.res3, decoder.res2, decoder.res1):
        siren_uniform_(res.conv1.weight, mode="fan_in", c=c_hidden)
        if res.conv1.bias is not None:
            nn.init.zeros_(res.conv1.bias)

        siren_uniform_(res.conv2.weight, mode="fan_in", c=6.0)
        with torch.no_grad():
            res.conv2.weight.mul_(0.1)
        if res.conv2.bias is not None:
            nn.init.zeros_(res.conv2.bias)

    # 4. Linear output head (unactivated)
    siren_uniform_(decoder.head.weight, mode="fan_in", c=6.0)
    if decoder.head.bias is not None:
        nn.init.zeros_(decoder.head.bias)


class AutoencoderMono(nn.Module):
    """
    Standard 1D CNN Autoencoder (GELU EncoderMono + GELU DecoderMono) used for Amplitude.
    Latent shape: (B, 14, 18) = 252 dims. Total parameters: 401,471.
    """

    def __init__(self):
        super().__init__()
        self.encoder = EncoderMono()
        self.decoder = DecoderMono()
        init_kaiming_weights(self)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out


class AutoencoderMonoSiren(nn.Module):
    """
    1D CNN Autoencoder pairing a GELU EncoderMono with a SIREN-activated DecoderSirenMono
    used for Phase reconstruction.
    Latent shape: (B, 14, 18) = 252 dims. Total parameters: 401,471.
    """

    def __init__(self, w0_initial: float = 1.0, w0: float = 1.0):
        super().__init__()
        self.encoder = EncoderMono()
        self.decoder = DecoderSirenMono(w0_initial=w0_initial, w0=w0)
        init_kaiming_weights(self.encoder)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out


# ==============================================================================
# SECTION 3: Loss, Training, Validation & Early Stopping
# ==============================================================================

class TimeWeightedSmoothL1Loss(nn.Module):
    def __init__(self, time_weight_np: np.ndarray, beta: float, device: torch.device, dtype: torch.dtype):
        super().__init__()
        w = np.asarray(time_weight_np, dtype=np.float64)
        w_mean = float(w.mean())
        if w_mean > 0.0:
            w = w / w_mean
        self.register_buffer("weight", torch.from_numpy(w).view(1, 1, -1).to(device=device, dtype=dtype))
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.loss_fn(pred, target) * self.weight).mean()


def train_one_epoch_mono(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    loader: DataLoader,
    clip_norm: float = 1.0,
    lambda_latent: float = 0.0,
) -> float:
    model.train(True)
    dtype = next(model.parameters()).dtype
    non_blocking = device.type == "cuda"
    total_loss = 0.0

    for batch in loader:
        inputs = batch.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        if lambda_latent > 0.0:
            outputs, z = model(inputs, return_latent=True)
            recon_loss = loss_fn(outputs, inputs)
            loss = recon_loss + lambda_latent * torch.mean(z ** 2)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

        loss.backward()

        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device = None,
    lambda_latent: float = 0.0,
) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    dtype = next(model.parameters()).dtype
    non_blocking = device.type == "cuda"
    running_vloss = 0.0

    with torch.no_grad():
        for vbatch in val_loader:
            weights = None
            if isinstance(vbatch, (list, tuple)):
                if len(vbatch) > 1:
                    vinputs, weights = vbatch[0], vbatch[1]
                    weights = weights.to(device=device, dtype=dtype, non_blocking=non_blocking)
                else:
                    vinputs = vbatch[0]
            else:
                vinputs = vbatch

            vinputs = vinputs.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if vinputs.ndim == 2:
                vinputs = vinputs.unsqueeze(1)

            if lambda_latent > 0.0:
                voutputs, z = model(vinputs, return_latent=True)
                recon_loss = loss_fn(voutputs, vinputs, weights) if weights is not None else loss_fn(voutputs, vinputs)
                vloss = recon_loss + lambda_latent * torch.mean(z ** 2)
            else:
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vinputs, weights) if weights is not None else loss_fn(voutputs, vinputs)
            running_vloss += vloss.item()

    return running_vloss / max(len(val_loader), 1)


class EarlyStopping:
    """
    Multi-criteria Early Stopping with NaN/divergence guards and min-LR floor patience.
    """

    def __init__(
        self,
        patience: int = 100,
        threshold: float = 0.005,
        min_lr: float = 5e-7,
        patience_at_min_lr: int = 50,
        mode: str = "rel",
        divergence_factor: float = 100.0,
        patience_divergence: int = 5,
        min_delta: float = None,
    ):
        self.patience = patience
        self.threshold = min_delta if min_delta is not None else threshold
        self.min_lr = min_lr
        self.patience_at_min_lr = patience_at_min_lr
        self.mode = mode.lower()
        self.divergence_factor = divergence_factor
        self.patience_divergence = patience_divergence

        self.counter = 0
        self.counter_at_min_lr = 0
        self.divergence_counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.stop_reason = ""

    def step(self, val_loss: float, current_lr: float = None) -> bool:
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.early_stop = True
            self.stop_reason = "Loss became NaN or Inf"
            return True

        if self.best_loss < float("inf") and val_loss > max(1.0, self.best_loss * self.divergence_factor):
            self.divergence_counter += 1
            if self.divergence_counter >= self.patience_divergence:
                self.early_stop = True
                self.stop_reason = (
                    f"Loss diverged ({val_loss:.2e} > {self.divergence_factor}x best {self.best_loss:.2e}) "
                    f"for {self.divergence_counter} consecutive epochs"
                )
                return True
        else:
            self.divergence_counter = 0

        if self.mode == "rel":
            improved = val_loss < self.best_loss * (1.0 - self.threshold)
        else:
            improved = val_loss < self.best_loss - self.threshold

        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.counter_at_min_lr = 0
            self.early_stop = False
            self.stop_reason = ""
        else:
            self.counter += 1
            if current_lr is not None and current_lr <= self.min_lr * 1.05:
                self.counter_at_min_lr += 1
                if self.counter_at_min_lr >= self.patience_at_min_lr:
                    self.early_stop = True
                    self.stop_reason = (
                        f"LR at floor ({current_lr:.2e} <= {self.min_lr:.2e}) "
                        f"without improvement for {self.counter_at_min_lr} epochs"
                    )
            if self.patience > 0 and self.counter >= self.patience:
                self.early_stop = True
                self.stop_reason = (
                    f"Global patience reached ({self.counter} epochs without "
                    f">={self.threshold * 100:.2f}% improvement)"
                )

        return self.early_stop


# ==============================================================================
# SECTION 4: Plotting Utilities
# ==============================================================================

def plot_training_curves(
    epochs,
    train_losses,
    val_losses,
    lrs,
    run_title: str,
    metric_name: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax_lr = ax.twinx()

    l1 = ax.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=1.1)
    l2 = ax.plot(epochs, val_losses, label="Validation Loss", color="#d62728", linewidth=1.1)
    l3 = ax_lr.plot(epochs, lrs, label="Learning Rate", color="black", linewidth=1.3, linestyle="--")

    ax.set_yscale("log")
    ax_lr.set_yscale("log")

    min_vloss_idx = int(np.argmin(val_losses))
    min_vloss = val_losses[min_vloss_idx]
    best_epoch = epochs[min_vloss_idx]

    ax.axhline(y=min_vloss, color="#d62728", linestyle=":", linewidth=1, alpha=0.85)
    ax.scatter([best_epoch], [min_vloss], color="#d62728", s=55, zorder=5)

    offset_y = min_vloss * 1.6 if min_vloss * 1.6 < max(val_losses) else min_vloss * 0.7
    ax.annotate(
        f"Min Val Loss: {min_vloss:.6e}\n(Epoch {best_epoch})",
        xy=(best_epoch, min_vloss),
        xytext=(best_epoch, offset_y),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
        bbox=dict(boxstyle="round,pad=0.35", edgecolor="#d62728", facecolor="#fff5f5", alpha=0.95),
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title(f"{run_title} - {metric_name} Training Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Loss (Log Scale)", fontsize=11, fontweight="semibold", color="#d62728")
    ax_lr.set_ylabel("Learning Rate (Log Scale)", fontsize=11, fontweight="semibold", color="black")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def plot_mismatch_distribution(mismatches: np.ndarray, run_title: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bins = np.logspace(-8, 0, 81)
    clipped_mismatches = np.clip(mismatches, 1e-8, 1.0)
    counts, _, _ = ax.hist(clipped_mismatches, bins=bins, color="#2b5c8f", edgecolor="#173559", alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlim(1e-8, 1.0)
    max_count = int(np.max(counts)) if len(counts) > 0 else 0
    ax.set_ylim(0, max(3000, int(max_count * 1.15)))

    ax.set_xlabel("Mismatch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mismatch distributions for the model-generated waveforms\n"
        f"GELU Amplitude + SIREN-decoder Phase reconstruction ({run_title})",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    med = np.median(mismatches)
    mean_val = np.mean(mismatches)
    max_val = np.max(mismatches)
    stats_text = f"Median: {med:.2e}\nMean: {mean_val:.2e}\nMax: {max_val:.2e}\nSamples: {len(mismatches)}"
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="#2b5c8f", facecolor="#f0f5fa", alpha=0.9),
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Mismatch distribution plot saved: {output_path}")


# ==============================================================================
# SECTION 5: CLI & Training Entrypoint
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-contained training: GELU AutoencoderMono (Amplitude) + SIREN-decoder AutoencoderMonoSiren (Phase)"
    )
    parser.add_argument("--title", type=str, default=None, help="Run title / prefix for output files")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs (default: 800)")
    parser.add_argument(
        "--small",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use small dataset variant (NRSur7dq4v2_dataset_small.hdf) (default: False)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable torch.compile (default: False)",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable TF32 precision on CUDA (default: False)",
    )
    parser.add_argument(
        "--f64",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable float64 (double) precision training (default: False)",
    )
    parser.add_argument(
        "--w0-initial",
        type=float,
        default=1.0,
        help="SIREN frequency factor w0 for first Phase decoder layer (default: 1.0)",
    )
    parser.add_argument(
        "--w0",
        type=float,
        default=1.0,
        help="SIREN frequency factor w0 for hidden Phase decoder layers (default: 1.0)",
    )
    return parser.parse_args()


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    min_lr: float,
    soap_eps: float,
    precondition_freq: int,
    warmup_steps: int,
    decay_steps: int,
):
    """
    Creates SOAP optimizer and Warmup + CosineAnnealingLR scheduler.
    For AutoencoderMonoSiren with w0 != 1.0 on hidden decoder layers, scales the hidden SIREN
    decoder LR by 1/w0 to keep pre-activation step sizes matched with the 1/w0 weight scale.
    """
    if isinstance(model, AutoencoderMonoSiren) and model.decoder.w0 != 1.0:
        w0_hidden = model.decoder.w0
        first_layer_ids = {id(p) for p in model.decoder.c6.parameters()}
        head_ids = {id(p) for p in model.decoder.head.parameters()}
        res_conv2_ids = {
            id(p)
            for res in (model.decoder.res3, model.decoder.res2, model.decoder.res1)
            for p in res.conv2.parameters()
        }
        unscaled_ids = first_layer_ids | head_ids | res_conv2_ids
        siren_hidden_params = [p for p in model.decoder.parameters() if id(p) not in unscaled_ids]
        base_params = list(model.encoder.parameters()) + [
            p for p in model.decoder.parameters() if id(p) in unscaled_ids
        ]
        param_groups = [
            {"params": base_params, "lr": lr},
            {"params": siren_hidden_params, "lr": lr / w0_hidden},
        ]
    else:
        param_groups = model.parameters()

    optimizer = opt_add.SOAP(
        param_groups,
        lr=lr,
        betas=(0.95, 0.95),
        shampoo_beta=0.99,
        weight_decay=0.0,
        precondition_frequency=precondition_freq,
        merge_dims=False,
        eps=soap_eps,
    )
    warmup_sched = lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    decay_sched = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=min_lr
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps]
    )
    return optimizer, scheduler


def main():
    args = parse_args()

    # Hardcoded Hyperparameters & Constants
    BATCH_SIZE = 128
    PRECONDITION_FREQ = 15
    BETA = 0.02
    CLIP_NORM = 1.0
    LR = 1e-3
    MIN_LR = 1e-8 if args.f64 else 1e-6
    SOAP_EPS = 1e-12 if args.f64 else 1e-8
    WARMUP_STEPS = 200
    EARLY_STOP = 0
    LAMBDA_LATENT = 0.0
    np_dtype = np.float64 if args.f64 else np.float32
    model_dtype = torch.float64 if args.f64 else torch.float32

    if args.f64:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("Precision: Float64 (double precision).")
    elif args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        print("Precision: TF32 enabled for CUDA matmul and cuDNN.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        print("Precision: Standard FP32 (highest precision, TF32 disabled).")

    title = args.title
    if not title:
        try:
            title = input("Enter title of this run: ").strip()
        except EOFError:
            title = ""
        if not title:
            title = f"AE_mono_siren_soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_name = f"Amplitude: AutoencoderMono (GELU) | Phase: AutoencoderMonoSiren(w0_init={args.w0_initial}, w0={args.w0})"
    print(
        f"=== Starting Run: {title} | {model_name} | "
        f"Optimizer: SOAP | Schedule: Warmup+Cosine | Epochs: {args.epochs} ==="
    )

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        print(
            f"Allocated GPU: {prop.name} | Compute Capability: {prop.major}.{prop.minor} | "
            f"VRAM: {prop.total_memory / (1024**3):.1f} GB"
        )
    else:
        print("Using CPU device.")

    # Load Dataset from HDF5
    filepath = "datasets/NRSur7dq4v2_dataset_small.hdf" if args.small else "datasets/NRSur7dq4v2_dataset.hdf"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'")

    print(f"Loading waveforms from '{filepath}' (small={args.small})...")
    with h5py.File(filepath, "r") as f:
        waveforms_ds = f["Waveforms"]
        total_avail, sample_len = waveforms_ds.shape
        if sample_len != 2048:
            raise ValueError(f"Expected waveform length 2048 for AutoencoderMonoSiren, got {sample_len}.")
        n_samples = total_avail
        amplitude = np.empty((n_samples, sample_len), dtype=np_dtype)
        phase = np.empty((n_samples, sample_len), dtype=np_dtype)
        chunk_size = 10000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            wf_chunk = waveforms_ds[start:end]
            amplitude[start:end] = np.abs(wf_chunk).astype(np_dtype)
            phase[start:end] = get_phase(wf_chunk).astype(np_dtype)

    print(f"Loaded {n_samples} waveforms (length: {sample_len}).")

    # Split Dataset (80% train / 10% val / 10% test)
    idx_all = np.arange(n_samples)
    gen = torch.Generator().manual_seed(42)
    j_train_idx, j_val_idx, j_test_idx = random_split(idx_all, [0.8, 0.1, 0.1], generator=gen)

    train_indices = np.array(j_train_idx.indices)
    val_indices = np.array(j_val_idx.indices)
    test_indices = np.array(j_test_idx.indices)
    print(f"Dataset split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Compute per-timestep mean and std strictly on the training set
    print("Normalizing Amplitude and Phase using train-set per-timestep statistics...")
    mu_ampl = np.zeros(sample_len, dtype=np.float64)
    sq_ampl = np.zeros(sample_len, dtype=np.float64)
    mu_phase = np.zeros(sample_len, dtype=np.float64)
    sq_phase = np.zeros(sample_len, dtype=np.float64)

    for s in range(0, len(train_indices), 5000):
        idx_blk = np.sort(train_indices[s:s + 5000])
        blk_a = amplitude[idx_blk].astype(np.float64)
        mu_ampl += blk_a.sum(axis=0)
        sq_ampl += (blk_a ** 2).sum(axis=0)

        blk_p = phase[idx_blk].astype(np.float64)
        mu_phase += blk_p.sum(axis=0)
        sq_phase += (blk_p ** 2).sum(axis=0)

    mu_ampl /= len(train_indices)
    sd_ampl = np.sqrt(np.maximum(sq_ampl / len(train_indices) - mu_ampl ** 2, 0.0)) + 1e-6

    mu_phase /= len(train_indices)
    sd_phase = np.sqrt(np.maximum(sq_phase / len(train_indices) - mu_phase ** 2, 0.0)) + 1e-6

    for s in range(0, n_samples, 5000):
        amplitude[s:s + 5000] = (amplitude[s:s + 5000] - mu_ampl) / sd_ampl
        phase[s:s + 5000] = (phase[s:s + 5000] - mu_phase) / sd_phase

    norm_npz_path = f"{title}_norm.npz"
    np.savez(
        norm_npz_path,
        mu_ampl=mu_ampl,
        sd_ampl=sd_ampl,
        mu_phase=mu_phase,
        sd_phase=sd_phase,
    )
    print(f"Saved normalization constants: {norm_npz_path}")

    ampl_tensor = torch.from_numpy(amplitude)
    phase_tensor = torch.from_numpy(phase)

    a_train = Subset(ampl_tensor, train_indices)
    p_train = Subset(phase_tensor, train_indices)
    a_val = Subset(ampl_tensor, val_indices)
    p_val = Subset(phase_tensor, val_indices)

    use_pin = device.type == "cuda"
    test_joint = TensorDataset(ampl_tensor[test_indices], phase_tensor[test_indices])
    test_loader = DataLoader(test_joint, batch_size=1024, shuffle=False, pin_memory=use_pin)

    a_train_loader = DataLoader(a_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_pin)
    a_val_loader = DataLoader(a_val, batch_size=2048, shuffle=False, pin_memory=use_pin)
    p_train_loader = DataLoader(p_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_pin)
    p_val_loader = DataLoader(p_val, batch_size=2048, shuffle=False, pin_memory=use_pin)

    # Models: standard GELU AutoencoderMono for Amplitude, SIREN-decoder AutoencoderMonoSiren for Phase
    raw_model_amplitude = AutoencoderMono().to(device=device, dtype=model_dtype)
    raw_model_phase = AutoencoderMonoSiren(
        w0_initial=args.w0_initial, w0=args.w0
    ).to(device=device, dtype=model_dtype)

    ampl_params = sum(p.numel() for p in raw_model_amplitude.parameters())
    phase_enc_params = sum(p.numel() for p in raw_model_phase.encoder.parameters())
    phase_dec_params = sum(p.numel() for p in raw_model_phase.decoder.parameters())
    print(
        f"Model parameters: Amplitude={ampl_params:,} (AutoencoderMono GELU) | "
        f"Phase={phase_enc_params + phase_dec_params:,} "
        f"(EncoderMono: {phase_enc_params:,}, DecoderSirenMono: {phase_dec_params:,})"
    )

    if args.compile:
        if sys.version_info >= (3, 13):
            print("Notice: torch.compile/Dynamo is not supported on Python 3.13+. Continuing with uncompiled models.")
            model_amplitude = raw_model_amplitude
            model_phase = raw_model_phase
        else:
            try:
                print("Compiling models with torch.compile...")
                model_amplitude = torch.compile(raw_model_amplitude)
                model_phase = torch.compile(raw_model_phase)
            except Exception as e:
                print(f"Notice: torch.compile failed ({e}). Continuing with uncompiled models.")
                model_amplitude = raw_model_amplitude
                model_phase = raw_model_phase
    else:
        print("torch.compile disabled. Running uncompiled eager models.")
        model_amplitude = raw_model_amplitude
        model_phase = raw_model_phase

    # Dedicated per-branch time-weighted SmoothL1 losses
    loss_fn_amplitude = TimeWeightedSmoothL1Loss(sd_ampl, beta=BETA, device=device, dtype=model_dtype)
    loss_fn_phase = TimeWeightedSmoothL1Loss(mu_ampl * sd_phase, beta=BETA, device=device, dtype=model_dtype)

    # Optimizers & Schedulers
    total_steps = args.epochs * max(len(a_train_loader), 1)
    warmup_steps = min(WARMUP_STEPS, max(1, total_steps // 2))
    decay_steps = max(total_steps - warmup_steps, 1)

    opt_fn_amplitude, scheduler_amplitude = build_optimizer_and_scheduler(
        raw_model_amplitude, LR, MIN_LR, SOAP_EPS, PRECONDITION_FREQ, warmup_steps, decay_steps
    )
    opt_fn_phase, scheduler_phase = build_optimizer_and_scheduler(
        raw_model_phase, LR, MIN_LR, SOAP_EPS, PRECONDITION_FREQ, warmup_steps, decay_steps
    )
    print(
        f"Schedule: {warmup_steps} warmup steps ({LR * 1e-3:.1e} -> {LR:.1e}) + "
        f"{decay_steps} cosine steps -> {MIN_LR:.1e} (Total: {total_steps} steps across {args.epochs} epochs)"
    )

    # Training Loop
    best_vloss_amplitude = float("inf")
    best_vloss_phase = float("inf")

    history = {
        "epoch": [],
        "train_loss_amplitude": [],
        "val_loss_amplitude": [],
        "lr_amplitude": [],
        "train_loss_phase": [],
        "val_loss_phase": [],
        "lr_phase": [],
        "epoch_time_sec": [],
    }

    stopped_amplitude = False
    stopped_phase = False

    if EARLY_STOP > 0:
        early_stop_amplitude = EarlyStopping(
            patience=EARLY_STOP, threshold=0.001, mode="rel", min_lr=MIN_LR, patience_at_min_lr=50
        )
        early_stop_phase = EarlyStopping(
            patience=EARLY_STOP, threshold=0.001, mode="rel", min_lr=MIN_LR, patience_at_min_lr=50
        )
        print(
            f"Early stopping enabled: patience={EARLY_STOP} "
            f"(gated to final 15%: epoch >= {int(0.85 * args.epochs)})"
        )
    else:
        print("Early stopping disabled.")

    print(
        f"Starting training for {args.epochs} epochs "
        f"(LR={LR:.1e}, clip_norm={CLIP_NORM}, beta={BETA}, batch_size={BATCH_SIZE}, "
        f"precond_freq={PRECONDITION_FREQ}, w0_initial={args.w0_initial}, w0={args.w0})..."
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss_ampl = train_one_epoch_mono(
            model_amplitude,
            opt_fn_amplitude,
            scheduler_amplitude,
            loss_fn_amplitude,
            device,
            a_train_loader,
            clip_norm=CLIP_NORM,
            lambda_latent=LAMBDA_LATENT,
        )
        avg_loss_ph = train_one_epoch_mono(
            model_phase,
            opt_fn_phase,
            scheduler_phase,
            loss_fn_phase,
            device,
            p_train_loader,
            clip_norm=CLIP_NORM,
            lambda_latent=LAMBDA_LATENT,
        )

        avg_vloss_ampl = validate(
            model_amplitude, a_val_loader, loss_fn_amplitude, device=device, lambda_latent=LAMBDA_LATENT
        )
        avg_vloss_ph = validate(
            model_phase, p_val_loader, loss_fn_phase, device=device, lambda_latent=LAMBDA_LATENT
        )

        current_lr_ampl = opt_fn_amplitude.param_groups[0]["lr"]
        current_lr_ph = opt_fn_phase.param_groups[0]["lr"]
        epoch_duration = time.time() - t0

        history["epoch"].append(epoch + 1)
        history["train_loss_amplitude"].append(avg_loss_ampl)
        history["val_loss_amplitude"].append(avg_vloss_ampl)
        history["lr_amplitude"].append(current_lr_ampl)
        history["train_loss_phase"].append(avg_loss_ph)
        history["val_loss_phase"].append(avg_vloss_ph)
        history["lr_phase"].append(current_lr_ph)
        history["epoch_time_sec"].append(epoch_duration)

        if avg_vloss_ampl < best_vloss_amplitude:
            best_vloss_amplitude = avg_vloss_ampl
            torch.save(raw_model_amplitude.state_dict(), f"{title}_amplitude_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model_amplitude.state_dict(),
                    "optimizer_state_dict": opt_fn_amplitude.state_dict(),
                    "scheduler_state_dict": scheduler_amplitude.state_dict(),
                    "best_vloss": best_vloss_amplitude,
                },
                f"{title}_amplitude_checkpoint.pt",
            )

        if avg_vloss_ph < best_vloss_phase:
            best_vloss_phase = avg_vloss_ph
            torch.save(raw_model_phase.state_dict(), f"{title}_phase_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model_phase.state_dict(),
                    "optimizer_state_dict": opt_fn_phase.state_dict(),
                    "scheduler_state_dict": scheduler_phase.state_dict(),
                    "best_vloss": best_vloss_phase,
                    "w0_initial": args.w0_initial,
                    "w0": args.w0,
                },
                f"{title}_phase_checkpoint.pt",
            )

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"Epoch [{epoch + 1}/{args.epochs}] ({epoch_duration:.2f}s)")
            print(
                f"  Amplitude | Train Loss: {avg_loss_ampl:.6f} | Val Loss: {avg_vloss_ampl:.6f} | "
                f"Best: {best_vloss_amplitude:.6f} | LR: {current_lr_ampl:.2e}"
            )
            print(
                f"  Phase     | Train Loss: {avg_loss_ph:.6f} | Val Loss: {avg_vloss_ph:.6f} | "
                f"Best: {best_vloss_phase:.6f} | LR: {current_lr_ph:.2e}"
            )

        if EARLY_STOP > 0 and epoch >= int(0.85 * args.epochs):
            if not stopped_amplitude:
                stopped_amplitude = early_stop_amplitude.step(avg_vloss_ampl, current_lr_ampl)
            if not stopped_phase:
                stopped_phase = early_stop_phase.step(avg_vloss_ph, current_lr_ph)

            if stopped_amplitude and stopped_phase:
                print(f"\n=== [Early Stopping Triggered at Epoch {epoch + 1}] ===")
                print(f"  Amplitude: {early_stop_amplitude.stop_reason}")
                print(f"  Phase:     {early_stop_phase.stop_reason}")
                break

    # Save Metrics (CSV and NPY)
    csv_filename = f"{title}_metrics.csv"
    try:
        df_history = pd.DataFrame(history)
        df_history.to_csv(csv_filename, index=False)
        print(f"\nUnified metrics CSV saved: {csv_filename}")
    except Exception as e:
        print(f"\nWarning: Failed to save metrics CSV ({e}).")

    npy_filename = f"{title}_metrics.npy"
    try:
        np.save(npy_filename, history)
        print(f"Unified metrics NPY saved: {npy_filename}")
    except Exception as e:
        print(f"\nWarning: Failed to save metrics NPY ({e}).")

    # Generate Plots
    plot_ampl_filename = f"{title}_amplitude_plot.png"
    plot_training_curves(
        epochs=history["epoch"],
        train_losses=history["train_loss_amplitude"],
        val_losses=history["val_loss_amplitude"],
        lrs=history["lr_amplitude"],
        run_title=title,
        metric_name="Amplitude",
        output_path=plot_ampl_filename,
    )

    plot_phase_filename = f"{title}_phase_plot.png"
    plot_training_curves(
        epochs=history["epoch"],
        train_losses=history["train_loss_phase"],
        val_losses=history["val_loss_phase"],
        lrs=history["lr_phase"],
        run_title=title,
        metric_name="Phase",
        output_path=plot_phase_filename,
    )

    print(f"\nTraining completed successfully for run '{title}'.")

    # Evaluate Best Saved Models and Plot Mismatch Distribution
    best_ampl_path = f"{title}_amplitude_best.pt"
    best_phase_path = f"{title}_phase_best.pt"

    print("\nLoading best saved models for mismatch evaluation...")
    raw_model_amplitude.load_state_dict(torch.load(best_ampl_path, map_location=device, weights_only=True))
    raw_model_phase.load_state_dict(torch.load(best_phase_path, map_location=device, weights_only=True))

    print("Computing test set waveform mismatches (unnormalized)...")
    norm_stats = {
        "mu_ampl": mu_ampl,
        "sd_ampl": sd_ampl,
        "mu_phase": mu_phase,
        "sd_phase": sd_phase,
    }
    mismatches = compute_mismatches(
        model_amplitude, model_phase, test_loader, device=device, norm=100.0, norm_stats=norm_stats
    )

    plot_mismatch_filename = f"{title}_mismatch_distribution.png"
    plot_mismatch_distribution(
        mismatches=mismatches,
        run_title=title,
        output_path=plot_mismatch_filename,
    )

    print("\nSaved artifacts:")
    print(f"  - Best Amplitude Model:      {best_ampl_path}")
    print(f"  - Amplitude Full Checkpoint: {title}_amplitude_checkpoint.pt (includes optimizer state)")
    print(f"  - Best Phase Model:          {best_phase_path}")
    print(f"  - Phase Full Checkpoint:     {title}_phase_checkpoint.pt (includes optimizer state)")
    print(f"  - Normalization NPZ (mu,sd): {norm_npz_path}")
    print(f"  - History CSV:               {csv_filename}")
    print(f"  - History NPY:               {npy_filename}")
    print(f"  - Amplitude Plot:            {plot_ampl_filename}")
    print(f"  - Phase Plot:                {plot_phase_filename}")
    print(f"  - Mismatch Plot:             {plot_mismatch_filename}")
    print(
        f"  - Test Set Mismatch Stats:   Median: {np.median(mismatches):.2e} | "
        f"Mean: {np.mean(mismatches):.2e} | Max: {np.max(mismatches):.2e}"
    )


if __name__ == "__main__":
    main()
