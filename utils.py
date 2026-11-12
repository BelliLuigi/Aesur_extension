import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pycbc import conversions

try:
    from gwsurrogate.new.precessing_surrogate import rotateWaveform, multiplyQuats
except ImportError:
    rotateWaveform = None
    multiplyQuats = None


# ==============================================================================
# SECTION 1: Signal Processing & Phase Alignment
# ==============================================================================

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


# ==============================================================================
# SECTION 2: Physics, Overlap & Inertial Waveforms
# ==============================================================================

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
    dtype = next(model_amplitude.parameters()).dtype

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

            true_wf = torch.polar(b_ampl.squeeze(1), -b_phase.squeeze(1) * norm)
            pred_wf = torch.polar(torch.clamp(pred_ampl.squeeze(1), min=0.0), -pred_phase.squeeze(1) * norm)

            match = TasosOverlapTorch(true_wf, pred_wf)
            mismatch = torch.clamp(1.0 - match, min=0.0)
            mismatches.extend(mismatch.cpu().numpy().tolist())

    return np.array(mismatches, dtype=np.float64)


def reconstruct_inertial_waveform(cooA, cooP, ophase, q, norm_phase: float = 100.0):
    """
    Reconstructs the inertial-frame (2, 2) waveform from co-orbital components
    following Section III G of Pürrer et al. (2026):
      1. Unscales orbital phase to physical radians: phi_orb = ophase * norm_phase
      2. Forms orbital rotation quaternion: R_z = (cos(phi/2), 0, 0, sin(phi/2))
      3. Composes full rotation quaternion: R = q_copr (x) R_z
      4. Projects R onto unit sphere
      5. Forms co-orbital modes array (m in [-2, ..., 2]) with h_{2,-2} = (h_{2,2})*
      6. Applies Wigner-D rotation to inertial frame: h_inert_22 = (rotateWaveform(R, h_modes))[4]

    Supports both single waveform (1D / 2D) and batched tensors/arrays.
    """
    global rotateWaveform, multiplyQuats
    if rotateWaveform is None or multiplyQuats is None:
        try:
            from gwsurrogate.new.precessing_surrogate import rotateWaveform, multiplyQuats
        except ImportError as e:
            raise ImportError(f"gwsurrogate is required for inertial waveform reconstruction: {e}")

    # Convert torch.Tensor to numpy
    if isinstance(cooA, torch.Tensor):
        cooA = cooA.detach().cpu().numpy()
    if isinstance(cooP, torch.Tensor):
        cooP = cooP.detach().cpu().numpy()
    if isinstance(ophase, torch.Tensor):
        ophase = ophase.detach().cpu().numpy()
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()

    # Squeeze channel dim if present
    if cooA.ndim == 3 and cooA.shape[1] == 1:
        cooA = cooA.squeeze(1)
    if cooP.ndim == 3 and cooP.shape[1] == 1:
        cooP = cooP.squeeze(1)
    if ophase.ndim == 3 and ophase.shape[1] == 1:
        ophase = ophase.squeeze(1)

    is_single = (cooA.ndim == 1)
    if is_single:
        cooA = cooA[np.newaxis, :]
        cooP = cooP[np.newaxis, :]
        ophase = ophase[np.newaxis, :]
        q = q[np.newaxis, ...]

    B, L = cooA.shape
    h_inert_22 = np.empty((B, L), dtype=np.complex128)

    for b in range(B):
        # 1. Unscale phase to radians
        phi_orb = ophase[b].astype(np.float64) * norm_phase

        # 2. Orbital phase rotation quaternion R_z(phi) = (cos(phi/2), 0, 0, sin(phi/2))
        q_rot = np.zeros((4, L), dtype=np.float64)
        q_rot[0] = np.cos(phi_orb / 2.0)
        q_rot[3] = np.sin(phi_orb / 2.0)

        # 3. Full rotation quaternion R = q_copr (x) R_z
        q_b = q[b].astype(np.float64)
        q_full = multiplyQuats(q_b, q_rot)

        # 4. Normalize full quaternion to unit sphere
        q_norm = np.linalg.norm(q_full, axis=0, keepdims=True) + 1e-12
        q_full = q_full / q_norm

        # 5. Co-orbital modes array (2, -2..2)
        # h_{2,2} = A * exp(-i * phase * norm_phase), h_{2,-2} = (h_{2,2})*
        h_22_coorb = np.clip(cooA[b], 0.0, None).astype(np.float64) * np.exp(
            -1j * cooP[b].astype(np.float64) * norm_phase
        )
        h_modes = np.zeros((5, L), dtype=np.complex128)
        h_modes[0] = np.conj(h_22_coorb)  # (2, -2)
        h_modes[4] = h_22_coorb           # (2, 2)

        # 6. Rotate to inertial frame
        h_inert_22[b] = rotateWaveform(q_full, h_modes)[4]

    return h_inert_22[0] if is_single else h_inert_22


def compute_inertial_mismatches(
    pred_cooA,
    pred_cooP,
    pred_ophase,
    pred_q,
    true_inertial,
    true_cooA=None,
    true_cooP=None,
    true_ophase=None,
    true_q=None,
    swap_component: str = None,
    norm_phase: float = 100.0,
):
    """
    Computes inertial-frame mismatches between reconstructed waveforms and true_inertial waveforms.
    Supports modular leave-one-out swapping:
      swap_component:
        - None: all predicted components
        - 'ophase': swap predicted ophase with true_ophase
        - 'q': swap predicted q with true_q
        - 'coo': swap predicted (cooA, cooP) with true_(cooA, cooP)
        - 'cooA': swap predicted cooA with true_cooA
        - 'cooP': swap predicted cooP with true_cooP
    """
    eff_cooA = true_cooA if (swap_component in ('coo', 'cooA') and true_cooA is not None) else pred_cooA
    eff_cooP = true_cooP if (swap_component in ('coo', 'cooP') and true_cooP is not None) else pred_cooP
    eff_ophase = true_ophase if (swap_component == 'ophase' and true_ophase is not None) else pred_ophase
    eff_q = true_q if (swap_component == 'q' and true_q is not None) else pred_q

    rec_inertial = reconstruct_inertial_waveform(
        eff_cooA, eff_cooP, eff_ophase, eff_q, norm_phase=norm_phase
    )

    if not isinstance(rec_inertial, torch.Tensor):
        rec_t = torch.from_numpy(rec_inertial)
    else:
        rec_t = rec_inertial

    if not isinstance(true_inertial, torch.Tensor):
        true_t = torch.from_numpy(true_inertial)
    else:
        true_t = true_inertial

    match = TasosOverlapTorch(true_t, rec_t)
    mismatch = torch.clamp(1.0 - match, min=0.0)
    return mismatch.cpu().numpy().astype(np.float64)


# ==============================================================================
# SECTION 3: Custom Loss Functions
# ==============================================================================

class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss supporting optional sample-level weights.
    Averages over the time/spatial dimensions first, then weights across the batch.
    """
    def __init__(self, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, pred, target, weights=None):
        elem_loss = self.loss_fn(pred, target)
        sample_loss = elem_loss.mean(dim=[-2, -1] if elem_loss.ndim == 3 else -1)
        if weights is not None:
            weights = weights.view_as(sample_loss)
            return torch.mean(sample_loss * weights)
        return torch.mean(sample_loss)


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


# ==============================================================================
# SECTION 4: Training & Validation Engine
# ==============================================================================

def train_one_epoch(model, optimizer, loss_fn, device, epoch_index, tb_writer, loader, lambda_latent: float = 0.0):
    model.train(True)
    dtype = next(model.parameters()).dtype
    total_loss = 0.0

    for i, batch in enumerate(loader):
        weights = None
        if isinstance(batch, (list, tuple)):
            if len(batch) > 1:
                inputs, weights = batch[0], batch[1]
                weights = weights.to(device=device, dtype=dtype)
            else:
                inputs = batch[0]
        else:
            inputs = batch
        inputs = inputs.to(device=device, dtype=dtype)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        if lambda_latent > 0.0:
            outputs, z = model(inputs, return_latent=True)
            recon_loss = loss_fn(outputs, inputs, weights) if weights is not None else loss_fn(outputs, inputs)
            l2_latent_reg = torch.mean(z ** 2)
            loss = recon_loss + lambda_latent * l2_latent_reg
        else:
            outputs = model(inputs)
            recon_loss = loss_fn(outputs, inputs, weights) if weights is not None else loss_fn(outputs, inputs)
            loss = recon_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if tb_writer is not None and i % 10 == 9:
            tb_x = epoch_index * len(loader) + i + 1
            tb_writer.add_scalar("Loss/train", loss.item(), tb_x)

    return total_loss / max(len(loader), 1)


def validate(model, val_loader, loss_fn, device=None, lambda_latent: float = 0.0):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    dtype = next(model.parameters()).dtype
    running_vloss = 0.0
    with torch.no_grad():
        for i, vbatch in enumerate(val_loader):
            weights = None
            if isinstance(vbatch, (list, tuple)):
                if len(vbatch) > 1:
                    vinputs, weights = vbatch[0], vbatch[1]
                    weights = weights.to(device=device, dtype=dtype)
                else:
                    vinputs = vbatch[0]
            else:
                vinputs = vbatch
            vinputs = vinputs.to(device=device, dtype=dtype)
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
    Multi-criteria Early Stopping aligned with ReduceLROnPlateau dynamics.
    Requires sustained divergence over multiple consecutive epochs to prevent false triggers.
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
        # 1. NaN / Inf guard (instant stop)
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.early_stop = True
            self.stop_reason = "Loss became NaN or Inf"
            return True

        # 2. Divergence guard: only trigger if loss stays > divergence_factor * best_loss for N epochs
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

        # 3. Check improvement
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

            # Condition A: LR at minimum floor without improvement
            if current_lr is not None and current_lr <= self.min_lr * 1.05:
                self.counter_at_min_lr += 1
                if self.counter_at_min_lr >= self.patience_at_min_lr:
                    self.early_stop = True
                    self.stop_reason = (
                        f"LR at floor ({current_lr:.2e} <= {self.min_lr:.2e}) "
                        f"without improvement for {self.counter_at_min_lr} epochs"
                    )

            # Condition B: Global patience exhausted
            if self.patience > 0 and self.counter >= self.patience:
                self.early_stop = True
                self.stop_reason = (
                    f"Global patience reached ({self.counter} epochs without "
                    f">={self.threshold * 100:.2f}% improvement)"
                )

        return self.early_stop


# ==============================================================================
# SECTION 5: Plotting, Diagnostics & Data Utilities
# ==============================================================================

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


def plot_coorbital_training_curves(history, component_names, run_title, output_path):
    """
    Plots a unified 2x3 grid of training curves for the 6 CoorbitalAE submodules:
    Each subplot shows Train Loss (Blue), Validation Loss (Red), Learning Rate (Black dashed, twinx),
    and marks the minimum validation loss with an annotated pointer.
    """
    epochs = history["epoch"]
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    axes_flat = axes.flatten()

    for idx, comp in enumerate(component_names):
        ax = axes_flat[idx]
        ax.set_facecolor("#ffffff")
        ax_lr = ax.twinx()

        train_l = history[f"train_loss_{comp}"]
        val_l = history[f"val_loss_{comp}"]
        lrs = history[f"lr_{comp}"]

        l1 = ax.plot(epochs, train_l, label="Train Loss", color="#1f77b4", linewidth=1.1)
        l2 = ax.plot(epochs, val_l, label="Val Loss", color="#d62728", linewidth=1.1)
        l3 = ax_lr.plot(epochs, lrs, label="LR", color="black", linewidth=1.1, linestyle="--")

        ax.set_yscale("log")
        ax_lr.set_yscale("log")

        # Minimum validation loss marker and annotation
        min_vloss_idx = int(np.argmin(val_l))
        min_vloss = val_l[min_vloss_idx]
        best_epoch = epochs[min_vloss_idx]

        ax.axhline(y=min_vloss, color="#d62728", linestyle=":", linewidth=0.9, alpha=0.85)
        ax.scatter([best_epoch], [min_vloss], color="#d62728", s=35, zorder=5)

        offset_y = min_vloss * 1.8 if min_vloss * 1.8 < max(val_l) else min_vloss * 0.6
        ax.annotate(
            f"Min: {min_vloss:.2e}\n(Ep {best_epoch})",
            xy=(best_epoch, min_vloss),
            xytext=(best_epoch, offset_y),
            arrowprops=dict(facecolor="black", shrink=0.08, width=0.8, headwidth=4),
            bbox=dict(boxstyle="round,pad=0.25", edgecolor="#d62728", facecolor="#fff5f5", alpha=0.9),
            fontsize=7.5,
            fontweight="bold",
        )

        ax.set_title(f"Submodule: {comp.upper()}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=9, fontweight="semibold")
        ax.set_ylabel("Loss (Log Scale)", fontsize=9, fontweight="semibold", color="#d62728")
        ax_lr.set_ylabel("LR", fontsize=9, fontweight="semibold", color="black")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8, loc="upper right")

    fig.suptitle(f"{run_title} - CoorbitalAE Training Curves (6 Submodules)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Unified 6-subplot training curves plot saved: {output_path}")


def plot_mismatch_distribution(mismatches, run_title: str, output_path: str):
    """
    Plots the mismatch distribution histogram:
      - X-axis: log scale from 10^-8 to 10^0 ("Mismatch")
      - Y-axis: counts from 0 to 4000 ("Count")
      - DPI: 300
    """
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
        f"Distribution for the autoencoder reconstruction ({run_title})",
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


def plot_inertial_mismatch_ablation(mismatches_dict, run_title: str, output_path: str):
    """
    Plots an overlay of inertial mismatch distributions comparing the full model against
    leave-one-out component replacements (inspired by Fig. 8 in Pürrer et al. 2026):
      - 'Full NN': Blue solid
      - 'ophase -> truth': Red solid
      - 'q -> truth': Green solid
      - 'coo -> truth': Orange solid
    """
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = {
        "Full NN": "#1f77b4",
        "ophase -> truth": "#d62728",
        "q -> truth": "#2ca02c",
        "coo -> truth": "#ff7f0e",
    }

    bins = np.logspace(-7, 0, 71)

    for name, mm in mismatches_dict.items():
        color = colors.get(name, "#444444")
        clipped = np.clip(mm, 1e-7, 1.0)
        med = float(np.median(mm))
        p95 = float(np.percentile(mm, 95))

        ax.hist(
            clipped,
            bins=bins,
            histtype="step",
            color=color,
            linewidth=1.7,
            label=f"{name} (med: {med:.1e}, p95: {p95:.1e})",
        )
        ax.axvline(med, color=color, linestyle=":", linewidth=1.1, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1.0)
    ax.set_xlabel("Inertial Waveform Mismatch (1 - Overlap)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Inertial Waveform Mismatch & Error Attribution (Leave-One-Out)\n{run_title}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Inertial mismatch ablation plot saved: {output_path}")


# --- Data Sampling & NR Loading Helpers ---

def get_NR_data(group, key):
    if key == "waveform":
        out = (
            group["waveform"]["(2,2)"][:]
            if "waveform" in group.keys()
            else group["Lev1/waveform"][:]
        )
        out = out[-2048:]
    elif key == "metadata":
        metagroup = "metadata"
        q = group[f"{metagroup}/mass_ratio"][()]
        s1z = group[f"{metagroup}/dimensionless_spin1"][2][()]
        s2z = group[f"{metagroup}/dimensionless_spin2"][2][()]
        s1y = group[f"{metagroup}/dimensionless_spin1"][1][()]
        s1x = group[f"{metagroup}/dimensionless_spin1"][0][()]
        s2y = group[f"{metagroup}/dimensionless_spin2"][1][()]
        s2x = group[f"{metagroup}/dimensionless_spin2"][0][()]
        ecc = group[f"{metagroup}/eccentricity"][()]
        m2 = 100 * q / (1 + q)
        m1 = 100 - m2
        chi_p = conversions.chi_p(m1, m2, s1x, s1y, s2x, s2y)
        out = np.array([q, s1z, s2z, chi_p, ecc])
    else:
        out = None
    return out


def farthest_point_sampling(pts: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Selects n_samples that maximally span the normalized 3D parameter space."""
    p_norm = (pts - pts.min(dim=0).values) / (pts.max(dim=0).values - pts.min(dim=0).values + 1e-8)
    selected = [0]
    dists = torch.cdist(p_norm, p_norm[0:1]).squeeze(1)
    for _ in range(1, n_samples):
        next_idx = torch.argmax(dists).item()
        selected.append(next_idx)
        dists = torch.minimum(dists, torch.cdist(p_norm, p_norm[next_idx : next_idx + 1]).squeeze(1))
    return torch.tensor(selected)
