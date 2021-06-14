import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_blocks import (
    DepthWiseConv, DepthWiseTransposedConv, DepthWiseSelfNormalizingSqueezeBlock
)

class STNVAE(nn.Module):
    "Super Tiny NVAE"
    def __init__(self):
        super(STNVAE, self).__init__()
        
        self.encoder_blocks = nn.Sequential(
            DepthWiseConv(1, 16, kernel_size=2, stride=2),
            nn.SELU(),
            DepthWiseSelfNormalizingSqueezeBlock(16, 16),
            nn.SELU(),
            DepthWiseConv(16, 32, kernel_size=2, stride=2),
            nn.SELU(),
            DepthWiseSelfNormalizingSqueezeBlock(32, 32),
        )

        self.decoder_blocks = nn.Sequential(
            DepthWiseSelfNormalizingSqueezeBlock(32, 32),
            nn.SELU(),
            DepthWiseTransposedConv(32, 16, kernel_size=2, stride=2),
            nn.SELU(),
            DepthWiseSelfNormalizingSqueezeBlock(16, 16),
            nn.SELU(),
            DepthWiseTransposedConv(16, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        z, kl_div = self.encode(x)
        return self.decode(z), kl_div

    def encode(self, x):
        z = self.encoder_blocks(x)
        return z, 0.5 * torch.sum(z.pow(2))

    def decode(self, z):
        return self.decoder_blocks(z)

if __name__ == "__main__":
    from conv_blocks import count_parameters
    x = torch.randn(3, 1, 28, 28).to(memory_format=torch.channels_last)
    z = torch.randn(3, 32, 7, 7).to(memory_format=torch.channels_last)
    model = STNVAE()
    print(count_parameters(model))
    print(model(x)[0].shape)
    print(model.decode(z).shape)