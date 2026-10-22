import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_phase(dataset):
    '''
    Inputs:
        dataset: complex dataset
    Outputs:
        phase: unwrapped and zero-aligned phase dataset scaled by 1/100
    '''
    phase_non = -1 * np.unwrap(np.angle(dataset)) / 100
    out = phase_non
    phase = out - out[..., np.newaxis, 0]
    return phase


def count_model_params(model: nn.Module) -> dict:
    """Computes trainable, non-trainable, and total parameters in PyTorch."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


def get_conv1d_params(in_length: int, out_length: int, in_channels: int, out_channels: int, bias: bool = True):
    """
    Finds valid (kernel_size, stride, padding) configurations to map in_length -> out_length,
    along with total learnable parameters and an instantiated nn.Conv1d layer.
    """
    candidates = []
    for stride in [1, 2, 4, 8]:
        for kernel_size in range(2, 50):
            for padding in range(kernel_size):
                num = in_length + 2 * padding - kernel_size
                if num >= 0 and num % stride == 0:
                    l_out = num // stride + 1
                    if l_out == out_length:
                        num_params = out_channels * in_channels * kernel_size + (out_channels if bias else 0)
                        candidates.append({
                            "kernel_size": kernel_size,
                            "stride": stride,
                            "padding": padding,
                            "trainable_params": num_params,
                            "layer": nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                        })
    return candidates


def get_conv_transpose1d_params(
    in_length: int,
    out_length: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
):
    """
    Finds valid (kernel_size, stride, padding) configurations to map in_length -> out_length,
    along with total learnable parameters and an instantiated nn.ConvTranspose1d layer.
    """
    candidates = []
    for stride in [1, 2, 4, 8]:
        for kernel_size in range(2, 50):
            for padding in range(kernel_size):
                l_out = (in_length - 1) * stride - 2 * padding + kernel_size
                if l_out == out_length:
                    num_params = in_channels * out_channels * kernel_size + (out_channels if bias else 0)
                    candidates.append({
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "padding": padding,
                        "trainable_params": num_params,
                        "layer": nn.ConvTranspose1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                    })
    return candidates


def train_one_epoch(model, optimizer, loss_fn, device, epoch_index, tb_writer, loader, lambda_latent: float = 0.0):
    model.train(True)
    total_loss = 0.0

    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        inputs = batch.float().to(device)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        if lambda_latent > 0.0:
            outputs, z = model(inputs, return_latent=True)
            recon_loss = loss_fn(outputs, inputs)
            l2_latent_reg = torch.mean(z ** 2)
            loss = recon_loss + lambda_latent * l2_latent_reg
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if tb_writer is not None and i % 10 == 9:
            tb_x = epoch_index * len(loader) + i + 1
            tb_writer.add_scalar("Loss/train", loss.item(), tb_x)

    return total_loss / max(len(loader), 1)


class L1WeightLoss(nn.Module):
    def __init__(self, w: float = 5.0, start_window: int = 1700, end_window: int = 2048):
        super().__init__()
        self.w = w
        self.start_window = start_window
        self.end_window = end_window
        weights = torch.ones(1, 1, self.end_window)
        weights[:, :, self.start_window:self.end_window] = torch.linspace(1.0, self.w, self.end_window - self.start_window)
        weights = weights / weights.mean()
        self.register_buffer("weights", weights)

    def forward(self, inputs, outputs):
        return torch.mean(self.weights * torch.abs(inputs - outputs))


def validate(model, val_loader, loss_fn, device=None, lambda_latent: float = 0.0):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vbatch in enumerate(val_loader):
            if isinstance(vbatch, (list, tuple)):
                vbatch = vbatch[0]
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
    return running_vloss / max(len(val_loader), 1)


def train_one_epoch_maesmatch(
    model_amplitude,
    model_phase,
    opt_amplitude,
    opt_phase,
    loss_fn_joint,
    loss_fn_ampl,
    loss_fn_phase,
    device,
    epoch_index,
    tb_writer,
    loader,
    lambda_latent: float = 0.0,
    lambda_joint: float = 0.1,
    norm: float = 100.0,
):
    """
    Trains amplitude and phase autoencoders jointly using:
      1) Dedicated L1WeightLoss on amplitude (breaks sign degeneracy)
      2) Dedicated L1WeightLoss on unwrapped phase (breaks 2*pi wrapping degeneracy)
      3) Strictly positive AESurLoss on the reconstructed complex waveform (maximizes GW overlap)
    Returns:
      (avg_ampl_total_loss, avg_phase_total_loss, avg_joint_waveform_loss)
    """
    model_amplitude.train(True)
    model_phase.train(True)

    total_ampl_loss = 0.0
    total_phase_loss = 0.0
    total_joint_loss = 0.0

    for i, (b_ampl, b_phase) in enumerate(loader):
        b_ampl = b_ampl.float().to(device)
        b_phase = b_phase.float().to(device)
        if b_ampl.ndim == 2:
            b_ampl = b_ampl.unsqueeze(1)
        if b_phase.ndim == 2:
            b_phase = b_phase.unsqueeze(1)

        opt_amplitude.zero_grad()
        opt_phase.zero_grad()

        if lambda_latent > 0.0:
            pred_ampl, z_ampl = model_amplitude(b_ampl, return_latent=True)
            pred_phase, z_phase = model_phase(b_phase, return_latent=True)
            reg_ampl = lambda_latent * torch.mean(z_ampl ** 2)
            reg_phase = lambda_latent * torch.mean(z_phase ** 2)
        else:
            pred_ampl = model_amplitude(b_ampl)
            pred_phase = model_phase(b_phase)
            reg_ampl = 0.0
            reg_phase = 0.0

        # 1. Dedicated direct reconstruction losses
        loss_ampl_direct = loss_fn_ampl(pred_ampl, b_ampl) + reg_ampl
        loss_phase_direct = loss_fn_phase(pred_phase, b_phase) + reg_phase

        # 2. Reconstruct polar complex waveforms: A * exp(-i * phase * norm)
        # Clamp amplitude >= 0 for polar conversion to prevent negative modulus flips
        true_wf = torch.polar(b_ampl.squeeze(1), -b_phase.squeeze(1) * norm)
        pred_wf = torch.polar(torch.clamp(pred_ampl.squeeze(1), min=0.0), -pred_phase.squeeze(1) * norm)

        loss_joint = loss_fn_joint(pred_wf, true_wf)

        # 3. Combined objective: each model gets its direct gradient + shared waveform overlap gradient
        loss = loss_ampl_direct + loss_phase_direct + lambda_joint * loss_joint

        loss.backward()
        opt_amplitude.step()
        opt_phase.step()

        # Track effective per-branch loss (direct + weighted joint)
        ampl_branch_loss = (loss_ampl_direct + lambda_joint * loss_joint).item()
        phase_branch_loss = (loss_phase_direct + lambda_joint * loss_joint).item()

        total_ampl_loss += ampl_branch_loss
        total_phase_loss += phase_branch_loss
        total_joint_loss += loss_joint.item()

        if tb_writer is not None and i % 10 == 9:
            tb_x = epoch_index * len(loader) + i + 1
            tb_writer.add_scalar("Loss/train_ampl_maesmatch", ampl_branch_loss, tb_x)
            tb_writer.add_scalar("Loss/train_phase_maesmatch", phase_branch_loss, tb_x)
            tb_writer.add_scalar("Loss/train_joint_only", loss_joint.item(), tb_x)

    n = max(len(loader), 1)
    return total_ampl_loss / n, total_phase_loss / n, total_joint_loss / n


def validate_maesmatch(
    model_amplitude,
    model_phase,
    val_loader,
    loss_fn_joint,
    loss_fn_ampl,
    loss_fn_phase,
    device=None,
    lambda_latent: float = 0.0,
    lambda_joint: float = 0.1,
    norm: float = 100.0,
):
    """
    Validates amplitude and phase autoencoders both individually and jointly on reconstructed complex waveforms.
    Returns:
      (avg_val_ampl_loss, avg_val_phase_loss, avg_val_joint_loss)
    """
    if device is None:
        device = next(model_amplitude.parameters()).device
    model_amplitude.eval()
    model_phase.eval()

    running_vloss_ampl = 0.0
    running_vloss_phase = 0.0
    running_vloss_joint = 0.0

    with torch.no_grad():
        for i, (vb_ampl, vb_phase) in enumerate(val_loader):
            vb_ampl = vb_ampl.float().to(device)
            vb_phase = vb_phase.float().to(device)
            if vb_ampl.ndim == 2:
                vb_ampl = vb_ampl.unsqueeze(1)
            if vb_phase.ndim == 2:
                vb_phase = vb_phase.unsqueeze(1)

            if lambda_latent > 0.0:
                pred_ampl, z_ampl = model_amplitude(vb_ampl, return_latent=True)
                pred_phase, z_phase = model_phase(vb_phase, return_latent=True)
                reg_ampl = lambda_latent * torch.mean(z_ampl ** 2)
                reg_phase = lambda_latent * torch.mean(z_phase ** 2)
            else:
                pred_ampl = model_amplitude(vb_ampl)
                pred_phase = model_phase(vb_phase)
                reg_ampl = 0.0
                reg_phase = 0.0

            vloss_ampl_direct = loss_fn_ampl(pred_ampl, vb_ampl) + reg_ampl
            vloss_phase_direct = loss_fn_phase(pred_phase, vb_phase) + reg_phase

            true_wf = torch.polar(vb_ampl.squeeze(1), -vb_phase.squeeze(1) * norm)
            pred_wf = torch.polar(torch.clamp(pred_ampl.squeeze(1), min=0.0), -pred_phase.squeeze(1) * norm)

            vloss_joint = loss_fn_joint(pred_wf, true_wf)

            running_vloss_ampl += (vloss_ampl_direct + lambda_joint * vloss_joint).item()
            running_vloss_phase += (vloss_phase_direct + lambda_joint * vloss_joint).item()
            running_vloss_joint += vloss_joint.item()

    n = max(len(val_loader), 1)
    return running_vloss_ampl / n, running_vloss_phase / n, running_vloss_joint / n


def plot_training_curves(epochs, train_losses, val_losses, lrs, run_title, metric_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Secondary y-axis for learning rate
    ax_lr = ax.twinx()

    # Curves: Blue (Train), Red (Validation) on primary axis
    l1 = ax.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=1.1)
    l2 = ax.plot(epochs, val_losses, label="Validation Loss", color="#d62728", linewidth=1.1)

    # Learning Rate (Black dashed) on secondary axis
    l3 = ax_lr.plot(epochs, lrs, label="Learning Rate", color="black", linewidth=1.3, linestyle="--")

    # Both y-axes are logarithmic
    ax.set_yscale("log")
    ax_lr.set_yscale("log")

    # Minimum validation loss marker and horizontal line
    min_vloss_idx = int(np.argmin(val_losses))
    min_vloss = val_losses[min_vloss_idx]
    best_epoch = epochs[min_vloss_idx]

    ax.axhline(y=min_vloss, color="#d62728", linestyle=":", linewidth=1, alpha=0.85)
    ax.scatter([best_epoch], [min_vloss], color="#d62728", s=55, zorder=5)

    # Offset tag annotation
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

    # Combine legends from both axes
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {output_path}")



def TasosOverlapTorch(h1, h2, dt=2 * 4.925794970773135e-06, eps: float = 1e-12):
    '''
    dt is 2M_odot in seconds.
    df is the frequency resolution.
    Promotes FFT and inner product to complex128/float64 so 1.0 - match does not
    suffer from float32 precision loss (~1e-7) when computing mismatches down to 1e-12.
    Output belongs to [-1, 1]: 1 is perfect match, 0 is orthogonal, -1 is phase-inverted.
    '''
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


def maesmatchLoss(true_waveforms, predicted_waveforms, eps: float = 1e-7):
    """
    Computes a strictly positive (>= 0) AESur composite loss between true and predicted waveforms
    using Option B (Soft-Log10 mismatch without waveform_mae):
        Loss = MAE(real) + MAE(imag) + log10(1 + (1 - Overlap) / eps)

    Parameters:
        true_waveforms: Ground truth waveform.
        predicted_waveforms: Reconstructed waveform in matching format.
        eps: Scale/smoothing factor (default 1e-7). Prevents gradient explosion at mismatch -> 0
             while preserving 1/mismatch log scaling for mismatch >> eps.

    Returns:
        torch.Tensor: Strictly non-negative scalar loss combining time-domain MAE and soft-log10 mismatch.
    """
    if torch.is_complex(true_waveforms):
        real_true, imag_true = true_waveforms.real, true_waveforms.imag
        real_pred, imag_pred = predicted_waveforms.real, predicted_waveforms.imag
    elif isinstance(true_waveforms, (tuple, list)):
        real_true, imag_true = true_waveforms
        real_pred, imag_pred = predicted_waveforms
    elif true_waveforms.ndim == 3 and true_waveforms.shape[1] == 2:
        real_true, imag_true = true_waveforms[:, 0], true_waveforms[:, 1]
        real_pred, imag_pred = predicted_waveforms[:, 0], predicted_waveforms[:, 1]
    else:
        real_true, imag_true = true_waveforms[0], true_waveforms[1]
        real_pred, imag_pred = predicted_waveforms[0], predicted_waveforms[1]

    # 1. Time-domain MAE on individual quadrature components (h+, hx)
    real_mae = torch.mean(torch.abs(real_true - real_pred))
    imag_mae = torch.mean(torch.abs(imag_true - imag_pred))

    # Form complex representations for frequency-domain overlap
    true_c = torch.complex(real_true, imag_true)
    pred_c = torch.complex(real_pred, imag_pred)

    # 2. Frequency-domain Soft-Log10 mismatch: log10(1 + (1 - match) / eps) >= 0
    match = TasosOverlapTorch(true_c, pred_c)
    mismatch = torch.clamp(1.0 - match, min=0.0)
    log_match = torch.mean(torch.log10(1.0 + mismatch / eps)).to(real_mae.dtype)

    loss = real_mae + imag_mae + log_match
    return loss


class AESurLoss(nn.Module):
    def __init__(self, dt: float = 2 * 4.925794970773135e-06, eps: float = 1e-7):
        super().__init__()
        self.dt = dt
        self.eps = eps

    def forward(self, predicted_waveforms, true_waveforms):
        return maesmatchLoss(true_waveforms, predicted_waveforms, eps=self.eps)


def compute_mismatches(model_amplitude, model_phase, loader, device=None, norm: float = 100.0):
    """
    Evaluates Amplitude and Phase Autoencoders over a DataLoader yielding (amplitude, phase),
    reconstructs physical polar waveforms h = A * exp(-i * phase * norm), and calculates
    individual waveform mismatches M = 1 - Overlap.
    """
    if device is None:
        device = next(model_amplitude.parameters()).device
    model_amplitude.eval()
    model_phase.eval()

    mismatches = []
    with torch.no_grad():
        for b_ampl, b_phase in loader:
            b_ampl = b_ampl.float().to(device)
            b_phase = b_phase.float().to(device)
            if b_ampl.ndim == 2:
                b_ampl = b_ampl.unsqueeze(1)
            if b_phase.ndim == 2:
                b_phase = b_phase.unsqueeze(1)

            pred_ampl = model_amplitude(b_ampl)
            pred_phase = model_phase(b_phase)

            true_wf = torch.polar(b_ampl.squeeze(1), -b_phase.squeeze(1) * norm)
            pred_wf = torch.polar(torch.clamp(pred_ampl.squeeze(1), min=0.0), -pred_phase.squeeze(1) * norm)

            match = TasosOverlapTorch(true_wf, pred_wf)
            mismatch = torch.clamp(1.0 - match, min=0.0)
            mismatches.extend(mismatch.cpu().numpy().tolist())

    return np.array(mismatches, dtype=np.float64)


def plot_mismatch_distribution(mismatches, run_title: str, output_path: str):
    """
    Plots the mismatch distribution histogram:
      - X-axis: log scale from 10^-8 to 10^0 ("Mismatch")
      - Y-axis: counts from 0 to 4000 ("Count")
      - DPI: 300
      - Title: "Mismatch distributions for the model-generated waveforms"
      - Subtitle: "Left: distribution for the autoencoder reconstruction of the NRHybSur3dq8"
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bins = np.logspace(-8, 0, 81)
    clipped_mismatches = np.clip(mismatches, 1e-8, 1.0)
    ax.hist(clipped_mismatches, bins=bins, color="#2b5c8f", edgecolor="#173559", alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlim(1e-8, 1.0)
    ax.set_ylim(0, 4000)

    ax.set_xlabel("Mismatch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Mismatch distributions for the model-generated waveforms\n"
        "Left: distribution for the autoencoder reconstruction of the NRHybSur3dq8",
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


class EarlyStopping:
    """
    Early stopping helper supporting relative ('rel') and absolute ('abs') improvement.
    Monitors validation loss and signals termination when improvement ceases or when
    the learning rate has reached its lower bound and progress halts.
    """
    def __init__(self, patience: int = 100, min_delta: float = 1e-3, mode: str = "rel", min_lr: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()  # "rel" (percentage, e.g. 1e-3 = 0.1%) or "abs"
        self.min_lr = min_lr
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss: float, current_lr: float = None) -> bool:
        if self.mode == "rel":
            improved = val_loss < self.best_loss * (1.0 - self.min_delta)
        else:
            improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # Stop if LR is at floor and loss has not improved for at least 20 epochs
        if current_lr is not None and current_lr <= self.min_lr and self.counter >= 20:
            self.early_stop = True

        return self.early_stop

