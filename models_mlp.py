"""MLP autoencoder and parameter-to-latent mapper modules in PyTorch.

Default sizes for ``input_dim=2048, latent_dim=24, labels=3`` (float32):
  * ``Encoder`` (``neurons_unit=128``): 2,790,296 params (10.64 MB)
  * ``Decoder`` (``neurons_unit=128``): 2,792,320 params (10.65 MB)
  * ``Full``    (``neurons_unit=64``):  2,893,016 params total (11.04 MB);
    mapper only = 100,696 params (0.38 MB) when ``trainable=False``.
"""

import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder: ``(B, input_dim) -> (B, latent_dim)``.

    Hidden widths: ``8n -> 4n -> 2n -> n -> latent_dim`` (GELU activations).
    Size (``neurons_unit=128``): ``8n*input_dim + 42n^2 + 15n + latent_dim*(n + 1)`` params.
    For ``input_dim=2048, latent_dim=24``: 2,790,296 params (10.64 MB in fp32).
    """

    def __init__(self, input_dim, latent_dim, neurons_unit=128):
        super().__init__()
        n = neurons_unit
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8 * n), nn.GELU(),
            nn.Linear(8 * n, 4 * n), nn.GELU(),
            nn.Linear(4 * n, 2 * n), nn.GELU(),
            nn.Linear(2 * n, n), nn.GELU(),
            nn.Linear(n, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """MLP decoder: ``(B, latent_dim) -> (B, input_dim)``.

    Hidden widths: ``n -> 2n -> 4n -> 8n -> input_dim`` (GELU activations).
    Size (``neurons_unit=128``): ``(8n + 1)*input_dim + 42n^2 + 15n + n*latent_dim`` params.
    For ``input_dim=2048, latent_dim=24``: 2,792,320 params (10.65 MB in fp32).
    """

    def __init__(self, input_dim, latent_dim, neurons_unit=128):
        super().__init__()
        n = neurons_unit
        self.net = nn.Sequential(
            nn.Linear(latent_dim, n), nn.GELU(),
            nn.Linear(n, 2 * n), nn.GELU(),
            nn.Linear(2 * n, 4 * n), nn.GELU(),
            nn.Linear(4 * n, 8 * n), nn.GELU(),
            nn.Linear(8 * n, input_dim),
        )

    def forward(self, z):
        return self.net(z)


class Full(nn.Module):
    """Parameter mapper + decoder: ``(B, labels) -> (B, latent_dim) -> (B, input_dim)``.

    Mapper widths: ``labels -> n -> 4n -> 4n -> n -> latent_dim``.
    Size (``labels=3, latent_dim=24, neurons_unit=64``):
      * Mapper: ``n*labels + 24n^2 + 10n + latent_dim*(n + 1)`` = 100,696 params (0.38 MB fp32).
      * Total (with ``Decoder(2048, 24)``): 2,893,016 params (11.04 MB fp32).
      * Trainable: 2,893,016 if ``trainable=True``, else 100,696.
    """

    def __init__(self, decoder, labels=3, latent_dim=24, neurons_unit=64, trainable=True):
        super().__init__()
        n = neurons_unit
        self.mapper = nn.Sequential(
            nn.Linear(labels, n),
            nn.Linear(n, 4 * n), nn.GELU(),
            nn.Linear(4 * n, 4 * n), nn.GELU(),
            nn.Linear(4 * n, n), nn.GELU(),
            nn.Linear(n, latent_dim),
        )
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.decoder(self.mapper(x))
