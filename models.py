import torch
import torch.nn as nn


class Encoder1(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        '''
        Docstring: 
        Convolutional layer bc they are leaner than FC layer and they keep temporal info better
        Batchnorm to stabilize training
        last layer before latent space is a fully connected, slope to be optuned
        Number of params: 3.5e4
        Number of params of fc AE: 3e6
        '''
        super().__init__()
        self.c1 = nn.Conv1d(in_channels= 1, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) #out lenght= 1024
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.c2 = nn.Conv1d(in_channels= 32, out_channels=16, kernel_size=40, stride=4, padding=18, bias=False) # o.l = 256
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.c3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=40, stride=4, padding=18, bias=False) # ol = 64
        self.bn3 = nn.BatchNorm1d(num_features=8) 
        self.fc = nn.Linear(in_features=512, out_features=64, bias = False)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self,x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        return self.fc(torch.flatten(x, start_dim=1))


class Decoder1(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(in_features=64, out_features=256, bias = False)
        self.bn0 = nn.BatchNorm1d(num_features=256)
        self.ct1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) #ol = 512
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.ct2 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) # ol = 1024
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.ct3 = nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=40, stride=2, padding=19, bias=False) #ol = 2048
        self.bn3 = nn.BatchNorm1d(num_features=8)
        self.ct4 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        x = self.act(self.bn0(self.fc(x)))
        x = x.unsqueeze(1)
        x = self.act(self.bn1(self.ct1(x)))
        x = self.act(self.bn2(self.ct2(x)))
        x = self.act(self.bn3(self.ct3(x)))
        return self.ct4(x)


class Autoencoder_CNN_1(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.encoder = Encoder1(negative_slope=negative_slope)
        self.decoder = Decoder1(negative_slope=negative_slope)
    def forward(self,x):
        return self.decoder(self.encoder(x))