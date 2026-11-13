import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil



class EncoderNOBN(nn.Module):
    """
    1D CNN Encoder for single-channel input without BatchNorm (bias=True).
    Dimensions: 2048 -> 1043 -> 530 -> 270 -> 138 -> 35 -> 64.
    Parameters: 145,344
    """
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels= 1, out_channels=16, kernel_size=40, stride=2, padding=38, padding_mode= "replicate") # out length = 1043
        self.c2 = nn.Conv1d(in_channels= 16, out_channels=32, kernel_size=20, stride=2, padding=18, padding_mode= "replicate") # out length = 530
        self.c3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=12, stride=2, padding=10, padding_mode= "replicate") # out length = 270
        self.c4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=6, padding_mode= "replicate") # out length = 138
        self.c5 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=4, stride=4, padding=2, padding_mode= "replicate") # out length = 35
        self.fc = nn.Linear(in_features=16 * 35, out_features=64)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.act(self.c5(x))
        return self.fc(torch.flatten(x, start_dim=1))


class DecoderNOBN(nn.Module):
    """
    1D CNN Decoder for single-channel output without BatchNorm (bias=True).
    Dimensions: 64 -> (16, 32) -> 128 -> 256 -> 512 -> 1024 -> 2048.
    Zero output_padding across all transposed convolutions.
    Parameters: 150,897
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=64, out_features=16 * 32)
        self.ct5 = nn.ConvTranspose1d(in_channels=16, out_channels=128, kernel_size=8, stride=4, padding=2) # out length = 128
        self.ct4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=8, stride=2, padding=3)  # out length = 256
        self.ct3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=12, stride=2, padding=5) # out length = 512
        self.ct2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=20, stride=2, padding=9) # out length = 1024
        self.ct1 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=40, stride=2, padding=19) # out length = 2048
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 32)
        x = self.act(self.ct5(x))
        x = self.act(self.ct4(x))
        x = self.act(self.ct3(x))
        x = self.act(self.ct2(x))
        return self.ct1(x)


class Autoencoder_CNN_NOBN(nn.Module):
    """
    1D CNN Autoencoder without BatchNorm (bias=True) and with zero output_padding in the decoder.
    Total parameters: 296,241 (EncoderNOBN: 145,344, DecoderNOBN: 150,897)
    """
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNOBN()
        self.decoder = DecoderNOBN()

    def forward(self, x, return_latent: bool = False):
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out



class EvenBiggerMapper(nn.Module):
    """
    MLP Parameter Mapper connecting physical parameters to latent representations.
    Architecture: 3 -> 64 -> 256 -> 256 -> 64.
    Parameters: 99,136
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=64)
        self.act = nn.GELU()
    
    def forward(self,x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)   


class FullModel(nn.Module):
    """
    Surrogate pipeline combining a parameter mapper and a waveform decoder.
    Input:  Physical parameters (B, num_params)
    Output: Reconstructed waveform (B, 2048)
    Total parameters: Dependent on submodules
        - e.g. EvenBiggerMapper (99,136) + DecoderNOBN (150,897) = 250,033
        - e.g. EvenBiggerMapper (99,136) + DecoderBig (284,969) = 384,105
        - e.g. EvenBiggerMapper (99,136) + DecoderW (174,081) = 273,217
    """
    def __init__(self, mapper: nn.Module, decoder: nn.Module):
        super().__init__()
        self.mapper = mapper
        self.decoder = decoder

    def forward(self, x):
        z = self.mapper(x)
        out = self.decoder(z)
        if out.ndim == 3 and out.size(1) == 1:
            out = out.squeeze(1)
        return out


class ResidualPhaseAutoencoder(nn.Module):
    """
    Phase Autoencoder with Mean-Trajectory Detrending (Residual Learning).
    Input:  phase (B, 1, 2048) or (B, 2048)
    Internal: delta_phase = phase - mean_phase (compresses target variance by >10x)
              out_residual = base_autoencoder(delta_phase)
    Output: reconstructed phase = out_residual + mean_phase
    Total parameters: Trainable parameters inherited from base_model
        - Base Autoencoder: 296,241 (if Autoencoder_CNN_NOBN)
        - Non-trainable buffer: mean_phase (2,048 values)
    """
    def __init__(self, base_model: nn.Module, mean_phase: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        if not isinstance(mean_phase, torch.Tensor):
            mean_phase = torch.as_tensor(mean_phase, dtype=torch.float32)
        self.register_buffer("mean_phase", mean_phase.view(1, 1, -1))

    def forward(self, phase, return_latent: bool = False):
        if phase.ndim == 2:
            phase = phase.unsqueeze(1)
        delta_phase = phase - self.mean_phase
        if return_latent:
            rec_res, z = self.base_model(delta_phase, return_latent=True)
            return rec_res + self.mean_phase, z
        rec_res = self.base_model(delta_phase)
        return rec_res + self.mean_phase


class ResBlock1D(nn.Module):
    """
    Intra-layer 1D residual block (x + F(x)) at fixed temporal resolution.
    Preserves boundary coverage via replicate padding.
    """
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding="same", padding_mode="replicate")
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding="same", padding_mode="replicate")
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class EncoderMono(nn.Module):
    """
    1D CNN Encoder with widened bottleneck (latent shape = (14, 18), dimension = 252) without BatchNorm (bias=True).
    Dimensions: 2048 -> 1043 -> 530 -> 270 -> 138 -> 35 -> 18.
    Channels: 1 -> 32 -> 32 -> 64 -> 128 -> 32 -> 14.
    Parameters: 130,350
    """
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=40, stride=2, padding=38, padding_mode="replicate") # out length = 1043
        self.c2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=20, stride=2, padding=18, padding_mode="replicate") # out length = 530
        self.c3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=12, stride=2, padding=10, padding_mode="replicate") # out length = 270
        self.c4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=6, padding_mode="replicate") # out length = 138
        self.c5 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4, stride=4, padding=2, padding_mode="replicate") # out length = 35
        self.c6 = nn.Conv1d(in_channels=32, out_channels=14, kernel_size=4, stride=2, padding=2, padding_mode="replicate") # out length = 18
        self.act = nn.GELU()

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.act(self.c5(x))
        return self.c6(x)


class DecoderMono(nn.Module):
    """
    1D CNN Decoder with widened bottleneck (latent shape = (14, 18), dimension = 252) without BatchNorm (bias=True).
    Dimensions: 18 -> 35 -> 138 -> 270 -> 530 -> 1043 -> 2048.
    Channels: 14 -> 32 -> 128 -> 64 -> 48 -> 32 -> 32 -> 1.
    Uses linear upsampling followed by standard Conv1d and ResBlock1D.
    Parameters: 271,121
    """
    def __init__(self):
        super().__init__()
        self.u6 = nn.Upsample(size=35, mode="linear", align_corners=False)
        self.c6 = nn.Conv1d(in_channels=14, out_channels=32, kernel_size=5, padding="same", padding_mode="replicate")
        self.u5 = nn.Upsample(size=138, mode="linear", align_corners=False)
        self.c5 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=7, padding="same", padding_mode="replicate")
        self.u4 = nn.Upsample(size=270, mode="linear", align_corners=False)
        self.c4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=9, padding="same", padding_mode="replicate")
        self.u3 = nn.Upsample(size=530, mode="linear", align_corners=False)
        self.c3 = nn.Conv1d(in_channels=64, out_channels=48, kernel_size=13, padding="same", padding_mode="replicate")
        self.res3 = ResBlock1D(channels=48, kernel_size=5)
        self.u2 = nn.Upsample(size=1043, mode="linear", align_corners=False)
        self.c2 = nn.Conv1d(in_channels=48, out_channels=32, kernel_size=21, padding="same", padding_mode="replicate")
        self.res2 = ResBlock1D(channels=32, kernel_size=7)
        self.u1 = nn.Upsample(size=2048, mode="linear", align_corners=False)
        self.c1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=41, padding="same", padding_mode="replicate")
        self.res1 = ResBlock1D(channels=32, kernel_size=7)
        self.head = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 14, 18)
        x = self.act(self.c6(self.u6(x)))
        x = self.act(self.c5(self.u5(x)))
        x = self.act(self.c4(self.u4(x)))
        x = self.res3(self.act(self.c3(self.u3(x))))
        x = self.res2(self.act(self.c2(self.u2(x))))
        x = self.res1(self.act(self.c1(self.u1(x))))
        return self.head(x)


class AutoencoderMono(nn.Module):
    """
    1D CNN Autoencoder with widened bottleneck (latent shape = (14, 18), dimension = 252)
    Total parameters: 401,471 (EncoderMono: 130,350, DecoderMono: 271,121)
    """
    def __init__(self):
        super().__init__()
        self.encoder = EncoderMono()
        self.decoder = DecoderMono()

    def forward(self, x, return_latent: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out


class EncoderW(nn.Module):
    """
    1D CNN Encoder with widened bottleneck (latent dimension = 256) without BatchNorm (bias=True).
    Dimensions: 2048 -> 1043 -> 530 -> 270 -> 138 -> 35 -> 256.
    Parameters: 253,056
    """
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=40, stride=2, padding=38, padding_mode="replicate")
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=20, stride=2, padding=18, padding_mode="replicate")
        self.c3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=12, stride=2, padding=10, padding_mode="replicate")
        self.c4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=6, padding_mode="replicate")
        self.c5 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=4, stride=4, padding=2, padding_mode="replicate")
        self.fc = nn.Linear(in_features=16 * 35, out_features=256)
        self.act = nn.GELU()

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.act(self.c5(x))
        return self.fc(torch.flatten(x, start_dim=1))


class DecoderW(nn.Module):
    """
    1D CNN Decoder with widened bottleneck (latent dimension = 256) without BatchNorm (bias=True).
    Dimensions: 256 -> (16, 32) -> 128 -> 256 -> 512 -> 1024 -> 2048.
    Zero output_padding across all transposed convolutions.
    Parameters: 249,201
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=256, out_features=16 * 32)
        self.ct5 = nn.ConvTranspose1d(in_channels=16, out_channels=128, kernel_size=8, stride=4, padding=2)
        self.ct4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.ct3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=12, stride=2, padding=5)
        self.ct2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=20, stride=2, padding=9)
        self.ct1 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=40, stride=2, padding=19)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 32)
        x = self.act(self.ct5(x))
        x = self.act(self.ct4(x))
        x = self.act(self.ct3(x))
        x = self.act(self.ct2(x))
        return self.ct1(x)


class AutoencoderW(nn.Module):
    """
    1D CNN Autoencoder with widened bottleneck (latent dimension = 256) without BatchNorm (bias=True).
    Total parameters: 502,257 (EncoderW: 253,056, DecoderW: 249,201)
    """
    def __init__(self):
        super().__init__()
        self.encoder = EncoderW()
        self.decoder = DecoderW()

    def forward(self, x, return_latent: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out



class FlexE(nn.Module):
    """
    Parametric 1D CNN Encoder with configurable bottleneck, width multiplier, and input channels.
    Dimensions: 2048 -> 1043 -> 530 -> 270 -> 138 -> 35 -> bottleneck.
    Parameters (default: bottleneck=64, width=1, channels=1): 44,460
    """
    def __init__(self, bottleneck=64, width=1, channels=1):
        super().__init__()
        wxc = int(ceil(width*channels))
        self.c1 = nn.Conv1d(in_channels= channels, out_channels=4*wxc, kernel_size=40, stride=2, padding=38, padding_mode= "replicate") # out length = 1043
        self.c2 = nn.Conv1d(in_channels= 4*wxc, out_channels=8*wxc, kernel_size=20, stride=2, padding=18, padding_mode= "replicate") # out length = 530
        self.c3 = nn.Conv1d(in_channels=8*wxc, out_channels=16*wxc, kernel_size=12, stride=2, padding=10, padding_mode= "replicate") # out length = 270
        self.c4 = nn.Conv1d(in_channels=16*wxc, out_channels=32*wxc, kernel_size=8, stride=2, padding=6, padding_mode= "replicate") # out length = 138
        self.c5 = nn.Conv1d(in_channels=32*wxc, out_channels=16, kernel_size=4, stride=4, padding=2, padding_mode= "replicate") # out length = 35
        self.fc = nn.Linear(in_features=16 * 35, out_features=bottleneck)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.act(self.c5(x))
        return self.fc(torch.flatten(x, start_dim=1))


class FlexD(nn.Module):
    """
    Parametric 1D CNN Decoder with configurable bottleneck, width multiplier, and output channels.
    Dimensions: bottleneck -> (16, 32) -> 128 -> 256 -> 512 -> 1024 -> 2048.
    Parameters (default: bottleneck=64, width=1, channels=1): 43,869
    """
    def __init__(self, bottleneck, width, channels=1):
        super().__init__()
        wxc = int(ceil(width*channels))
        self.fc = nn.Linear(in_features=bottleneck, out_features=16 * 32)
        self.ct5 = nn.ConvTranspose1d(in_channels=16, out_channels=32*wxc, kernel_size=8, stride=4, padding=2) # out length = 128
        self.ct4 = nn.ConvTranspose1d(in_channels=32*wxc, out_channels=16*wxc, kernel_size=8, stride=2, padding=3)  # out length = 256
        self.ct3 = nn.ConvTranspose1d(in_channels=16*wxc, out_channels=8*wxc, kernel_size=12, stride=2, padding=5) # out length = 512
        self.ct2 = nn.ConvTranspose1d(in_channels=8*wxc, out_channels=4*wxc, kernel_size=20, stride=2, padding=9) # out length = 1024
        self.ct1 = nn.ConvTranspose1d(in_channels=4*wxc, out_channels=channels, kernel_size=40, stride=2, padding=19) # out length = 2048
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 32)
        x = self.act(self.ct5(x))
        x = self.act(self.ct4(x))
        x = self.act(self.ct3(x))
        x = self.act(self.ct2(x))
        return self.ct1(x)




class CoorbitalAE(nn.Module):
    """
    Modular 1D CNN Autoencoder decomposing precessing waveforms into co-orbital kinematics and dynamics.
    Total parameters: 861,021 across 12 submodules (6 encoders + 6 decoders):
        - ophase (bottleneck=8,  width=0.5, ch=1): E_ophase (13,044)  + D_ophase (15,197)  = 28,241
        - q      (bottleneck=48, width=1.0, ch=4): E_q (138,288)     + D_q (144,628)      = 282,916
        - chi1   (bottleneck=48, width=1.0, ch=3): E_chi1 (91,156)   + D_chi1 (95,447)    = 186,603
        - chi2   (bottleneck=64, width=1.0, ch=3): E_chi2 (100,132)  + D_chi2 (103,639)   = 203,771
        - cooA   (bottleneck=48, width=1.0, ch=1): E_cooA (35,484)   + D_cooA (35,677)    = 71,161
        - cooP   (bottleneck=64, width=1.0, ch=1): E_cooP (44,460)   + D_cooP (43,869)    = 88,329
    """
    def __init__(self):
        super().__init__()
        self.E_ophase = FlexE(bottleneck=8, width=0.5, channels=1)
        self.D_ophase = FlexD(bottleneck=8, width=0.5, channels=1)
        self.E_q = FlexE(bottleneck=48, width=1, channels=4)
        self.D_q = FlexD(bottleneck=48, width=1, channels=4)
        self.E_chi1 = FlexE(bottleneck=48, width=1, channels=3)
        self.D_chi1 =  FlexD(bottleneck=48, width=1, channels=3)
        self.E_chi2 =  FlexE(bottleneck=64, width=1, channels=3)
        self.D_chi2 =  FlexD(bottleneck=64, width=1, channels=3)
        self.E_cooA =  FlexE(bottleneck=48, width=1, channels=1)
        self.D_cooA =  FlexD(bottleneck=48, width=1, channels=1)
        self.E_cooP =  FlexE(bottleneck=64, width=1, channels=1)
        self.D_cooP =   FlexD(bottleneck=64, width=1, channels=1)
    
    def forward(self, ophase, q, chi1, chi2, cooA, cooP, return_latent: bool = False):
        z_ophase = self.E_ophase(ophase)
        z_q = self.E_q(q)
        z_chi1 = self.E_chi1(chi1)
        z_chi2 = self.E_chi2(chi2)
        z_cooA = self.E_cooA(cooA)
        z_cooP = self.E_cooP(cooP)

        rec_ophase = self.D_ophase(z_ophase)
        rec_q = F.normalize(self.D_q(z_q))
        rec_chi1 = self.D_chi1(z_chi1)
        rec_chi2 = self.D_chi2(z_chi2)
        rec_cooA = F.relu(self.D_cooA(z_cooA))
        rec_cooP = self.D_cooP(z_cooP)

        recs = (rec_ophase, rec_q, rec_chi1, rec_chi2, rec_cooA, rec_cooP)
        if return_latent:
            latents = (z_ophase, z_q, z_chi1, z_chi2, z_cooA, z_cooP)
            return recs, latents
        return recs
