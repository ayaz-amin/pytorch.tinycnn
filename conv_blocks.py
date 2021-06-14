import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SqueezeBlock(nn.Module):
    '''
    SqueezeNext: Hardware-Aware Neural Network Design
    https://arxiv.org/abs/1803.10615

    This is a SqueezeNext block. It is essentially a ResBlock
    with a smaller parameter count.
    '''
    def __init__(self, in_channels, out_channels):
        super(SqueezeBlock, self).__init__()
        self.fwd_model = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels * 0.5), kernel_size=1),
            nn.BatchNorm2d(int(in_channels * 0.5)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.25), kernel_size=1),
            nn.BatchNorm2d(int(in_channels * 0.25)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_channels * 0.25), int(in_channels * 0.5), kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(in_channels * 0.5)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(int(in_channels * 0.5)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_channels * 0.5), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        if out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        z = self.fwd_model(x) + self.shortcut(x)
        return F.relu(z, inplace=True)

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, kaiming=True):
        super(DepthWiseConv, self).__init__()
        self.dwc = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
        if kaiming:
            nn.init.kaiming_normal_(self.dwc.weight, mode='fan_in', nonlinearity='linear') 
            nn.init.kaiming_normal_(self.pwc.weight, mode='fan_in', nonlinearity='linear')   
    
    def forward(self, x):
        return self.pwc(self.dwc(x))

class DepthWiseTransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, kaiming=True):
        super(DepthWiseTransposedConv, self).__init__()
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.dwc = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_channels,
            bias=bias
        )

        if kaiming:
            nn.init.kaiming_normal_(self.pwc.weight, mode='fan_in', nonlinearity='linear')
            nn.init.kaiming_normal_(self.dwc.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x):
        return self.dwc(self.pwc(x))

class DepthWiseSqueezeBlock(nn.Module):
    '''
    Based on:

    SqueezeNext: Hardware-Aware Neural Network Design
    https://arxiv.org/abs/1803.10615

    This is a Depthwise SqueezeNext block. It is more memory efficient lol.
    '''
    def __init__(self, in_channels, out_channels):
        super(DepthWiseSqueezeBlock, self).__init__()
        self.fwd_model = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels * 0.5), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(in_channels * 0.5)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.25), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(in_channels * 0.25)),
            nn.ReLU(inplace=True),

            DepthWiseConv(int(in_channels * 0.25), int(in_channels * 0.5), kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(in_channels * 0.5)),
            nn.ReLU(inplace=True),

            DepthWiseConv(int(in_channels * 0.5), int(in_channels * 0.5), kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(int(in_channels * 0.5)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_channels * 0.5), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        if out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        z = self.fwd_model(x) + self.shortcut(x)
        return F.relu(z, inplace=True)

class SelfNormalizingSqueezeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfNormalizingSqueezeBlock, self).__init__()
        '''
        SqueezeBlock based on the following papers:

        SqueezeNext: Hardware-Aware Neural Network Design
        https://arxiv.org/abs/1803.10615

        SNDCNN: SELF-NORMALIZING DEEP CNNs WITH SCALED EXPONENTIAL LINEAR UNITS FOR SPEECH RECOGNITION
        https://arxiv.org/abs/1910.01992

        This is a self normalizing SqueezeNext block. It has less parameters and
        is more memory efficient due to the lack of batch normalizing layers and
        residual connections.
        '''

        self.fwd_model = nn.Sequential(
            self.conv(in_channels, int(in_channels * 0.5), kernel_size=1, bias=False),
            nn.SELU(inplace=True),
            
            self.conv(int(in_channels * 0.5), int(in_channels * 0.25), kernel_size=1, bias=False),
            nn.SELU(inplace=True),

            self.conv(int(in_channels * 0.25), int(in_channels * 0.5), kernel_size=(3, 1), padding=(1, 0)),
            nn.SELU(inplace=True),

            self.conv(int(in_channels * 0.5), int(in_channels * 0.5), kernel_size=(1, 3), padding=(0, 1)),
            nn.SELU(inplace=True),

            self.conv(int(in_channels * 0.5), out_channels, kernel_size=1, bias=False),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        return self.fwd_model(x)

    def conv(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias
        )
        nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='linear')
        return conv

class DepthWiseSelfNormalizingSqueezeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseSelfNormalizingSqueezeBlock, self).__init__()
        '''
        SqueezeBlock based on the following papers:

        SqueezeNext: Hardware-Aware Neural Network Design
        https://arxiv.org/abs/1803.10615

        SNDCNN: SELF-NORMALIZING DEEP CNNs WITH SCALED EXPONENTIAL LINEAR UNITS FOR SPEECH RECOGNITION
        https://arxiv.org/abs/1910.01992

        This is a depthwise self normalizing SqueezeNext block. It has even fewer
        parameters and is even more memory efficient due to the lack of batch normalizing layers and
        residual connections, and presence of hybrid depthwise + pointwise separable convolutions
        '''

        self.fwd_model = nn.Sequential(
            self.conv(in_channels, int(in_channels * 0.5), kernel_size=1, bias=False),
            nn.SELU(inplace=True),
            
            self.conv(int(in_channels * 0.5), int(in_channels * 0.25), kernel_size=1, bias=False),
            nn.SELU(inplace=True),

            DepthWiseConv(int(in_channels * 0.25), int(in_channels * 0.5), kernel_size=(3, 1), padding=(1, 0)),
            nn.SELU(inplace=True),

            DepthWiseConv(int(in_channels * 0.5), int(in_channels * 0.5), kernel_size=(1, 3), padding=(0, 1)),
            nn.SELU(inplace=True),

            self.conv(int(in_channels * 0.5), out_channels, kernel_size=1, bias=False),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        return self.fwd_model(x)

    def conv(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias
        )
        nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='linear')
        return conv

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    channels_sizes = {
        "16x16": (16, 16),
        "10x32": (16, 32),
        "32x32": (32, 32),
        "32x64": (32, 64),
        "64x64": (64, 64)
    }

    x = np.array([0, 1, 2, 3, 4])
    x_ticks = ["16x16", "16x32", "32x32", "32x64", "64x64"]

    pcount_sb = []
    pcount_dwsb = []
    pcount_conv = []
    pcount_dwc = []
    pcount_snsb = []
    pcount_dwsnsb = []

    for _, channels in channels_sizes.items():
        pcount_sb.append(
            count_parameters(SqueezeBlock(channels[0], channels[1]))
        )
        pcount_dwsb.append(
            count_parameters(DepthWiseSqueezeBlock(channels[0], channels[1]))
        )
        pcount_conv.append(
            count_parameters(nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1))
        )
        pcount_dwc.append(
            count_parameters(DepthWiseConv(channels[0], channels[1], kernel_size=3, padding=1))
        )
        pcount_snsb.append(
            count_parameters(SelfNormalizingSqueezeBlock(channels[0], channels[1]))
        )
        pcount_dwsnsb.append(
            count_parameters(DepthWiseSelfNormalizingSqueezeBlock(channels[0], channels[1]))
        )

    pcount_sb = np.array(pcount_sb)
    pcount_dwsb = np.array(pcount_dwsb)
    pcount_conv = np.array(pcount_conv)
    pcount_dwc = np.array(pcount_dwc)
    pcount_snsb = np.array(pcount_snsb)
    pcount_dwsnsb = np.array(pcount_dwsnsb)

    plt.xticks(x, x_ticks)
    plt.xlabel("Channel shapes")
    plt.ylabel("Parameter count")

    plt.plot(x, pcount_sb, label="SqueezeBlock")
    plt.plot(x, pcount_dwsb, label="Depthwise SqueezeBlock (ours)")
    plt.plot(x, pcount_conv, label="Vanilla Convolution")
    plt.plot(x, pcount_dwc, label="Depthwise Convolution")
    plt.plot(x, pcount_snsb, label="Self normalizing SqueezeBlock (ours)")
    plt.plot(x, pcount_dwsnsb, label="Depthwise self normalizing SqueezeBlock (ours)")

    plt.legend()
    plt.show()