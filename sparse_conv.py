import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.2):
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

        input_size = in_channels * self.dwc.kernel_size[0] * self.dwc.kernel_size[1]
        num_nz = int(round((1 - sparsity) * input_size))
        zero_mask = torch.ones(in_channels, input_size, dtype=torch.bool)

        for in_channel in range(in_channels):
            in_idx = torch.multinomial(
                torch.ones(input_size),
                num_samples=num_nz,
                replacement=False
            )

            zero_mask[in_channel, in_idx] = False

        zero_mask = zero_mask.view(in_channels, in_channels, *self.dwc.kernel_size)
        self.dwc.weight.data[zero_mask] = 0

        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        return self.pwc(self.dwc(x))

if __name__ == "__main__":
    from conv_blocks import count_parameters
    
    x = torch.randn(3, 1, 28, 28)

    vconv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
    dwconv = DepthWiseConv(1, 3, kernel_size=3, padding=1)
    print(count_parameters(vconv))
    print(count_parameters(dwconv))
    print(vconv(x).shape)
    print(dwconv(x).shape)