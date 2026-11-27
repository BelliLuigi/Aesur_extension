"""MLP autoencoder and parameter-to-latent mapper modules in PyTorch.

Default sizes for ``input_dim=2048, latent_dim=24, labels=3`` (float32):
  * ``Encoder``        (``neurons_unit=128``): 2,790,296 params (10.64 MB)
  * ``Decoder``        (``neurons_unit=128``): 2,792,320 params (10.65 MB)
  * ``Full``           (``neurons_unit=64``):  2,893,016 params total (11.04 MB);
    mapper only = 100,696 params (0.38 MB) when ``trainable=False``.
  * ``NarrowEncoder``  (``n=128``):            708,632 params (2.70 MB)
  * ``LinearMatrices`` (``layers=4``):         2,304 params (0.01 MB; 576/layer)
  * ``NarrowDecoder``  (``n=128``):            710,656 params (2.71 MB)
  * ``irmae``          (narrow + 4 linear):    1,421,592 params (5.42 MB)
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
            nn.Linear(input_dim, 8 * n), nn.GELU(), ## 2048 -> 1024
            nn.Linear(8 * n, 4 * n), nn.GELU(), # 1024 -> 512
            nn.Linear(4 * n, 2 * n), nn.GELU(), # 512 -> 256
            nn.Linear(2 * n, n), nn.GELU(), #256 -> 128
            nn.Linear(n, latent_dim), # 128 ->24
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



class NarrowEncoder(nn.Module):
    """Narrow MLP encoder: ``(B, input_dim) -> (B, latent_dim)``.

    Hidden widths: ``2n -> 2n -> 2n -> n -> n -> latent_dim`` (GELU activations).
    Size: ``2n*input_dim + 11n^2 + 8n + latent_dim*(n + 1)`` params.
    For ``input_dim=2048, latent_dim=24, n=128``: 708,632 params (2.70 MB in fp32).
    """

    def __init__(self, input_dim, latent_dim, n=128):
        super().__init__()
        self.input_l = nn.Linear(input_dim, 2 * n)
        self.h1 = nn.Linear(2 * n, 2 * n)
        self.h2 = nn.Linear(2 * n, 2 * n)
        self.h3 = nn.Linear(2 * n, n)
        self.h4 = nn.Linear(n, n)
        self.zE = nn.Linear(n, latent_dim)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for layer in (self.input_l, self.h1, self.h2, self.h3, self.h4):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.zE.weight)
        nn.init.zeros_(self.zE.bias)

    def forward(self, x):
        x = self.act(self.input_l(x))
        x = self.act(self.h1(x))
        x = self.act(self.h2(x))
        x = self.act(self.h3(x))
        x = self.act(self.h4(x))
        return self.zE(x)


class LinearMatrices(nn.Module):
    """Stack of ``layers`` bias-free linear latent transforms: ``(B, latent_dim) -> (B, latent_dim)``.

    Size: ``layers * latent_dim^2`` params (PyTorch default init per IRMAE paper).
    For ``latent_dim=24``: 576 params/layer (``layers=4``: 2,304 params, 0.01 MB in fp32).
    """

    def __init__(self, latent_dim, layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NarrowDecoder(nn.Module):
    """Narrow MLP decoder: ``(B, latent_dim) -> (B, input_dim)``.

    Hidden widths: ``n -> n -> 2n -> 2n -> 2n -> input_dim`` (GELU activations).
    Size: ``(2n + 1)*input_dim + 11n^2 + 8n + n*latent_dim`` params.
    For ``input_dim=2048, latent_dim=24, n=128``: 710,656 params (2.71 MB in fp32).
    """

    def __init__(self, input_dim, latent_dim, n=128):
        super().__init__()
        self.zD = nn.Linear(latent_dim, n)
        self.h1 = nn.Linear(n, n)
        self.h2 = nn.Linear(n, 2 * n)
        self.h3 = nn.Linear(2 * n, 2 * n)
        self.h4 = nn.Linear(2 * n, 2 * n)
        self.output_l = nn.Linear(2 * n, input_dim)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for layer in (self.zD, self.h1, self.h2, self.h3, self.h4):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_l.weight)
        nn.init.zeros_(self.output_l.bias)

    def forward(self, z):
        z = self.act(self.zD(z))
        z = self.act(self.h1(z))
        z = self.act(self.h2(z))
        z = self.act(self.h3(z))
        z = self.act(self.h4(z))
        return self.output_l(z)


class irmae(nn.Module):
    """Implicit Rank-Minimizing Autoencoder (IRMAE): ``encoder -> linear_matrices -> decoder``.

    Size: ``params(encoder) + params(linear_matrices) + params(decoder)``.
    With ``NarrowEncoder(2048, 24, n=128)``, ``LinearMatrices(24, layers=4)``,
    and ``NarrowDecoder(2048, 24, n=128)``: 1,421,592 params (5.42 MB in fp32).
    """

    def __init__(self, encoder, linear_matrices, decoder):
        super().__init__()
        self.e = encoder
        self.lm = linear_matrices
        self.d = decoder

    def forward(self, x):
        return self.d(self.lm(self.e(x))) 
