#!/usr/bin/env python3
"""
PyTorch modules for the complex128 POD + Autoencoder + Mapper surrogate pipeline.

Architecture summary:
  - PODAutoencoder:   512 -> 256 -> 128 -> 64 -> 128 -> 256 -> 512 (GELU hidden activations)
  - PODMapper:        7 -> 16 -> 32 -> 64 -> 64 (GELU hidden activations)
  - PODReconstructor: 512 real [c_real, c_imag] -> 2048 complex waveform (initialized from POD_basis)
  - PODFullModel:     params (B, 7) -> PODMapper -> PODDecoder -> PODReconstructor -> waveform (B, 2048)
"""
import h5py
import numpy as np
import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Return total (or trainable-only) parameter count of a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def coeffs_complex_to_real(coeffs: torch.Tensor) -> torch.Tensor:
    """Convert complex POD coefficients (..., k) to stacked real tensor (..., 2*k)."""
    return torch.cat([coeffs.real, coeffs.imag], dim=-1)


def coeffs_real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert stacked real tensor (..., 2*k) back to complex POD coefficients (..., k)."""
    k = x.shape[-1] // 2
    return torch.complex(x[..., :k], x[..., k:])


class PODEncoder(nn.Module):
    """
    MLP Encoder mapping stacked real/imag POD coefficients to latent space.
    Default: 512 -> 256 -> 128 -> 64 with GELU activations.
    Parameters: 172,480
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: tuple[int, ...] = (256, 128),
        latent_dim: int = 64,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, dtype=dtype))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            x = coeffs_complex_to_real(x)
        return self.net(x)


class PODDecoder(nn.Module):
    """
    MLP Decoder mapping latent vectors back to stacked real/imag POD coefficients.
    Default: 64 -> 128 -> 256 -> 512 with GELU hidden activations.
    Parameters: 172,928
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (128, 256),
        out_dim: int = 512,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, dtype=dtype))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, out_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PODAutoencoder(nn.Module):
    """
    Symmetric MLP Autoencoder on stacked real/imag POD coefficients.
    Default: 512 -> 256 -> 128 -> 64 -> 128 -> 256 -> 512 (GELU activations).
    Parameters: 345,408 (PODEncoder: 172,480 + PODDecoder: 172,928)
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: tuple[int, ...] = (256, 128),
        latent_dim: int = 64,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.encoder = PODEncoder(in_dim=in_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, dtype=dtype)
        self.decoder = PODDecoder(
            latent_dim=latent_dim,
            hidden_dims=tuple(reversed(hidden_dims)),
            out_dim=in_dim,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out


class PODMapper(nn.Module):
    """
    MLP mapping 7D physical parameters (q, chiA, chiB) to the 64D AE latent space.
    Default: 7 -> 16 -> 32 -> 64 -> 64 with GELU hidden activations.
    Parameters: 6,944
    """

    def __init__(
        self,
        in_params: int = 7,
        hidden_dims: tuple[int, ...] = (32, 128, 128),
        latent_dim: int = 64,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_params
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, dtype=dtype))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)


class PODReconstructor(nn.Module):
    """
    Differentiable POD basis layer converting predicted coefficients (B, 2*k)
    back into complex time-domain waveforms (B, L): h_hat = c @ V_k + mean_wf.

    Stores real and imaginary parts of V_k as real PyTorch Parameters so standard
    real-valued optimizers (e.g. SOAP, Adam) can keep them frozen during AE
    pretraining and optionally unfreeze them during SXS finetuning.
    Parameters (k=256, L=2048): 1,048,576 (basis_real + basis_imag; frozen by default)
        + 4,096 non-trainable buffer values (mean_real + mean_imag)
    """

    def __init__(
        self,
        pod_basis: np.ndarray | torch.Tensor,
        pod_mean: np.ndarray | torch.Tensor | None = None,
        trainable: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        basis_t = torch.as_tensor(pod_basis)
        if basis_t.ndim != 2 or not torch.is_complex(basis_t):
            raise ValueError(f"pod_basis must be complex 2D (k, L), got shape={basis_t.shape}, dtype={basis_t.dtype}")

        self.k, self.length = basis_t.shape
        self.basis_real = nn.Parameter(basis_t.real.to(dtype=dtype).clone(), requires_grad=trainable)
        self.basis_imag = nn.Parameter(basis_t.imag.to(dtype=dtype).clone(), requires_grad=trainable)

        if pod_mean is None:
            mean_r = torch.zeros((1, self.length), dtype=dtype)
            mean_i = torch.zeros((1, self.length), dtype=dtype)
        else:
            mean_t = torch.as_tensor(pod_mean).reshape(1, self.length)
            mean_r = mean_t.real.to(dtype=dtype).clone()
            mean_i = mean_t.imag.to(dtype=dtype).clone()

        self.register_buffer("mean_real", mean_r)
        self.register_buffer("mean_imag", mean_i)

    def freeze(self):
        """Lock the POD basis weights (use during surrogate AE pretraining)."""
        self.basis_real.requires_grad_(False)
        self.basis_imag.requires_grad_(False)
        return self

    def unfreeze(self):
        """Make the POD basis weights trainable (use during SXS finetuning)."""
        self.basis_real.requires_grad_(True)
        self.basis_imag.requires_grad_(True)
        return self

    @classmethod
    def from_hdf5(cls, hdf_path: str, trainable: bool = False, dtype: torch.dtype = torch.float64):
        """Load POD_basis and POD_mean from a file written by pod_complex_save.py."""
        with h5py.File(hdf_path, "r") as f:
            pod_basis = f["POD_basis"][:]
            pod_mean = f["POD_mean"][:] if "POD_mean" in f else None
        return cls(pod_basis=pod_basis, pod_mean=pod_mean, trainable=trainable, dtype=dtype)

    def forward(self, coeffs_real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs_real: real tensor (B, 2*k) = [c_real, c_imag], or complex (B, k)
        Returns:
            Complex reconstructed waveform tensor of shape (B, L)
        """
        if torch.is_complex(coeffs_real):
            c_r, c_i = coeffs_real.real, coeffs_real.imag
        else:
            c_r, c_i = coeffs_real[..., : self.k], coeffs_real[..., self.k :]

        # (c_r + i c_i) @ (V_r + i V_i) = (c_r V_r - c_i V_i) + i (c_r V_i + c_i V_r)
        h_r = c_r @ self.basis_real - c_i @ self.basis_imag + self.mean_real
        h_i = c_r @ self.basis_imag + c_i @ self.basis_real + self.mean_imag
        return torch.complex(h_r, h_i)


class PODFullModel(nn.Module):
    """
    End-to-end waveform surrogate:
      params (B, 7) -> PODMapper -> latent z (B, 64) -> PODDecoder -> c_hat (B, 512) -> PODReconstructor -> h_hat (B, 2048)
    Parameters (default k=256, L=2048):
      - Trainable (frozen basis): 179,872 (PODMapper: 6,944 + PODDecoder: 172,928)
      - Total (including PODReconstructor basis): 1,228,448
    """

    def __init__(self, mapper: PODMapper, decoder: PODDecoder, reconstructor: PODReconstructor):
        super().__init__()
        self.mapper = mapper
        self.decoder = decoder
        self.reconstructor = reconstructor

    def forward(self, params: torch.Tensor, return_coeffs: bool = False):
        z = self.mapper(params)
        c_hat = self.decoder(z)
        h_hat = self.reconstructor(c_hat)
        if return_coeffs:
            return h_hat, c_hat
        return h_hat
