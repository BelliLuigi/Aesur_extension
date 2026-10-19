import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_phase(dataset):
    '''
    Inputs:
        dataset: complex dataset
    Outputs:
        ampl: amplitude dataset
        phase: phase dataset
    '''
    phase_non = -1*np.unwrap(np.angle(dataset))/100
    out = phase_non
    phase= out-out[..., np.newaxis,0]
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
    running_loss = 0.0
    last_loss = 0.0

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

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10
            #print(f"  batch {i + 1} loss: {last_loss:.6f}")
            if tb_writer is not None:
                tb_x = epoch_index * len(loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                if lambda_latent > 0.0:
                    tb_writer.add_scalar("Loss/recon_train", recon_loss.item(), tb_x)
                    tb_writer.add_scalar("Loss/latent_l2_train", l2_latent_reg.item(), tb_x)
            running_loss = 0.0

    return last_loss



class L1WeightLoss(nn.Module):
    def __init__(self, w:float = 5.0, start_window:int = 1700, end_window:int = 2048):
        super().__init__()
        self.w = w
        self.start_window = start_window
        self.end_window = end_window
        weights = torch.ones(1,1,self.end_window)
        weights[:,:, self.start_window:self.end_window] = torch.linspace(1.0, self.w, self.end_window - self.start_window)
        weights = weights/weights.mean()
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
            vinputs = vbatch.float().to(device)
            if vinputs.ndim == 2:
                vinputs = vinputs.unsqueeze(1)
            if lambda_latent > 0.0:
                voutputs, z = model(vinputs, return_latent=True)
                vloss = loss_fn(voutputs, vinputs) + lambda_latent * torch.mean(z**2)
            else:
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vinputs)
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


def plot_training_curves(epochs, train_losses, val_losses, lrs, run_title, metric_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Curves: Blue (Train), Red (Validation), Black (Learning Rate)
    ax.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=1.8)
    ax.plot(epochs, val_losses, label="Validation Loss", color="red", linewidth=1.8)
    ax.plot(epochs, lrs, label="Learning Rate", color="black", linewidth=1.4, linestyle="--")

    # Logarithmic y-scale
    ax.set_yscale("log")

    # Minimum validation loss marker and horizontal line
    min_vloss_idx = int(np.argmin(val_losses))
    min_vloss = val_losses[min_vloss_idx]
    best_epoch = epochs[min_vloss_idx]

    ax.axhline(y=min_vloss, color="red", linestyle=":", linewidth=1.5, alpha=0.85)
    ax.scatter([best_epoch], [min_vloss], color="red", s=55, zorder=5)

    # Offset tag annotation
    offset_y = min_vloss * 1.6 if min_vloss * 1.6 < max(val_losses) else min_vloss * 0.7
    ax.annotate(
        f"Min Val Loss: {min_vloss:.6e}\n(Epoch {best_epoch})",
        xy=(best_epoch, min_vloss),
        xytext=(best_epoch, offset_y),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
        bbox=dict(boxstyle="round,pad=0.35", edgecolor="red", facecolor="#fff5f5", alpha=0.95),
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title(f"{run_title} - {metric_name} Training Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Loss / Learning Rate (Log Scale)", fontsize=11, fontweight="semibold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {output_path}")


# Alias
plot_training = plot_training_curves