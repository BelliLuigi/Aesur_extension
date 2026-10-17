import torch
import torch.nn as nn


class Encoder1_old(nn.Module):
    """
    1D CNN Encoder for single-channel input.
    Parameters: 59,760
    """
    def __init__(self, negative_slope: float = 0.2):
        '''
        Docstring: 
        Convolutional layer bc they are leaner than FC layer and they keep temporal info better
        Batchnorm to stabilize training
        last layer before latent space is a fully connected, slope to be optuned
        Number of params: 59,760 (~5.98e4)
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


class Encoder1(nn.Module):
    """
    1D CNN Encoder for single-channel input.
    Parameters: 59,760
    """
    def __init__(self):
        '''
        Docstring: 
        Convolutional layer bc they are leaner than FC layer and they keep temporal info better
        Batchnorm to stabilize training
        last layer before latent space is a fully connected, slope to be optuned
        Number of params: 59,760 (~5.98e4)
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
        self.act = nn.GELU()

    def forward(self,x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        return self.fc(torch.flatten(x, start_dim=1))

class Decoder1(nn.Module):
    """
    1D CNN Decoder for single-channel output.
    Parameters: 69,576
    """
    def __init__(self):
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
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn0(self.fc(x)))
        x = x.unsqueeze(1)
        x = self.act(self.bn1(self.ct1(x)))
        x = self.act(self.bn2(self.ct2(x)))
        x = self.act(self.bn3(self.ct3(x)))
        return self.ct4(x)


class Decoder1_old(nn.Module):
    """
    1D CNN Decoder for single-channel output.
    Parameters: 69,576
    """
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
    """
    1D CNN Autoencoder for single-channel signals (Encoder1 + Decoder1).
    Total parameters: 129,336 (Encoder1: 59,760, Decoder1: 69,576)
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


### Double features AE

class Encoder2_old(nn.Module):
    """
    1D CNN Encoder for 2-channel input (amplitude & phase).
    Parameters: 173,280
    """
    def __init__(self, negative_slope: float = 0.2):
        '''
        So joint AutoEncoder that takes in input Ampl and phase together as 2 channels
        Number of params: 173,280 (~1.73e5)
        '''
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=40, stride=2, padding=19, bias=False)# ol = 1024
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.c2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=40, stride=4, padding=18, bias=False) # ol = 256
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.c3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=40, stride=4, padding=18, bias=False) # ol = 64
        self.bn3 = nn.BatchNorm1d(num_features=16)
        self.fc = nn.Linear(in_features=1024, out_features=64, bias=False)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self,x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        return self.fc(torch.flatten(x, start_dim=1))

class Decoder2_old(nn.Module):
    """
    1D CNN Decoder for 2-channel output (amplitude & phase).
    Parameters: 87,808
    """
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(in_features=64, out_features=512, bias = False)
        self.bn0 = nn.BatchNorm1d(num_features=512)
        self.ct1 = nn.ConvTranspose1d(in_channels=2, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) #ol = 512
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.ct2 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) # ol = 1024
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.ct3 = nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=40, stride=2, padding=19, bias=False) #ol = 2048
        self.bn3 = nn.BatchNorm1d(num_features=8)
        self.ct4 = nn.Conv1d(in_channels=8, out_channels=2, kernel_size=7, stride=1, padding=3, bias=False)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        x = self.act(self.bn0(self.fc(x))).view(-1, 2, 256)
        x = self.act(self.bn1(self.ct1(x)))
        x = self.act(self.bn2(self.ct2(x)))
        x = self.act(self.bn3(self.ct3(x)))
        return self.ct4(x)

class Decoder2(nn.Module):
    """
    1D CNN Decoder for 2-channel output (amplitude & phase).
    Parameters: 87,808
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=64, out_features=512, bias = False)
        self.bn0 = nn.BatchNorm1d(num_features=512)
        self.ct1 = nn.ConvTranspose1d(in_channels=2, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) #ol = 512
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.ct2 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=40, stride=2, padding=19, bias=False) # ol = 1024
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.ct3 = nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=40, stride=2, padding=19, bias=False) #ol = 2048
        self.bn3 = nn.BatchNorm1d(num_features=8)
        self.ct4 = nn.Conv1d(in_channels=8, out_channels=2, kernel_size=7, stride=1, padding=3, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn0(self.fc(x))).view(-1, 2, 256)
        x = self.act(self.bn1(self.ct1(x)))
        x = self.act(self.bn2(self.ct2(x)))
        x = self.act(self.bn3(self.ct3(x)))
        return self.ct4(x)

class Encoder2(nn.Module):
    """
    1D CNN Encoder for 2-channel input (amplitude & phase).
    Parameters: 173,280
    """
    def __init__(self):
        '''
        So joint AutoEncoder that takes in input Ampl and phase together as 2 channels
        Number of params: 173,280 (~1.73e5)
        '''
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=40, stride=2, padding=19, bias=False)# ol = 1024
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.c2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=40, stride=4, padding=18, bias=False) # ol = 256
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.c3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=40, stride=4, padding=18, bias=False) # ol = 64
        self.bn3 = nn.BatchNorm1d(num_features=16)
        self.fc = nn.Linear(in_features=1024, out_features=64, bias=False)
        self.act = nn.GELU()
    
    def forward(self,x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        x = self.act(self.bn3(self.c3(x)))
        return self.fc(torch.flatten(x, start_dim=1))


class Autoencoder_CNN_2(nn.Module):
    """
    1D CNN Autoencoder for 2-channel signals (Encoder2 + Decoder2).
    Total parameters: 261,088 (Encoder2: 173,280, Decoder2: 87,808)
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2()

    def forward(self, x, return_latent: bool = False):
        z = self.encoder(x)
        out = self.decoder(z)
        if return_latent:
            return out, z
        return out