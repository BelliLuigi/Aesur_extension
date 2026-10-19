import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder1(nn.Module):
    """
    1D CNN Encoder for single-channel input.
    Parameters: 145,536
    """
    def __init__(self):
        '''
        Docstring: 
        Convolutional layer bc they are leaner than FC layer and they keep temporal info better
        Batchnorm to stabilize training
        last layer before latent space is a fully connected, slope to be optuned
        Number of params: 145,536 (~1.46e5)
        Number of params of fc AE: 3e6
        '''
        super().__init__()
        self.c1 = nn.Conv1d(in_channels= 1, out_channels=16, kernel_size=40, stride=2, padding=38, bias=False, padding_mode= "replicate") # out length = 1043
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.c2 = nn.Conv1d(in_channels= 16, out_channels=32, kernel_size=20, stride=2, padding=18, bias=False, padding_mode= "replicate") # out length = 530
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.c3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=12, stride=2, padding=10, bias=False, padding_mode= "replicate") # out length = 270
        self.bn3 = nn.BatchNorm1d(num_features=64) 
        self.c4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=6, bias=False, padding_mode= "replicate") # out length = 138
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.c5 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=4, stride=4, padding=2, bias=False, padding_mode= "replicate") # out length = 35
        self.bn5 = nn.BatchNorm1d(num_features=16) 
        self.fc = nn.Linear(in_features=16 * 35, out_features=64, bias=False)
        self.act = nn.GELU()

    def forward(self,x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        x = self.act(self.bn4(self.c4(x)))
        x = self.act(self.bn5(self.c5(x)))
        return self.fc(torch.flatten(x, start_dim=1))

class Decoder1(nn.Module):
    """
    1D CNN Decoder for single-channel output.
    Parameters: 145,536
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=64, out_features=16 * 35, bias=False)
        self.bn0 = nn.BatchNorm1d(num_features=16)
        self.ct5 = nn.ConvTranspose1d(in_channels=16, out_channels=128, kernel_size=4, stride=4, padding=2, output_padding=2, bias=False) # out length = 138
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.ct4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=8, stride=2, padding=6, bias=False) # out length = 270
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.ct3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=12, stride=2, padding=10, bias=False) # out length = 530
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.ct2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=20, stride=2, padding=18, output_padding=1, bias=False) # out length = 1043
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.ct1 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=40, stride=2, padding=38, bias=False) # out length = 2048
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 35)
        x = self.act(self.bn0(x))
        x = self.act(self.bn5(self.ct5(x)))
        x = self.act(self.bn4(self.ct4(x)))
        x = self.act(self.bn3(self.ct3(x)))
        x = self.act(self.bn2(self.ct2(x)))
        return self.ct1(x)


class Autoencoder_CNN_1(nn.Module):
    """
    1D CNN Autoencoder for single-channel signals (Encoder1 + Decoder1).
    Total parameters: 291,072 (Encoder1: 145,536, Decoder1: 145,536)
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder1()
        self.decoder = Decoder1()

    def forward(self, x, return_latent: bool = False):
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out
