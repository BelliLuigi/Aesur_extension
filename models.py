import torch
import torch.nn as nn
import torch.nn.functional as F



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






