import numpy as np
import torch
import torch.nn as nn


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