import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class FSRCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        outer_channels=30,
        inner_channels=10,
        maps=5
    ):
        super().__init__()
        self.bicubic = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.extract = ConvBlock(
            in_channels,
            out_channels=outer_channels,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.shrink = ConvBlock(
            in_channels=outer_channels,
            out_channels=inner_channels,
            kernel_size=1
        )
        self.map = nn.Sequential(
            *[ConvBlock(
                in_channels=inner_channels,
                out_channels=inner_channels,
                kernel_size=3,
                padding=1,
                stride=1
            ) for _ in range(maps)]
        )
        self.expand = ConvBlock(
            in_channels=inner_channels,
            out_channels=outer_channels,
            kernel_size=1
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=outer_channels,
            out_channels=in_channels,
            kernel_size=9,
            stride=4,
            padding=3,
            output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=inner_channels,
            out_channels=in_channels,
            kernel_size=9,
            stride=2,
            padding=4,
            output_padding=1
        )

    def forward(self, x):
        bicubic = self.bicubic(x)
        first = self.extract(x)
        mid = self.expand(self.map(self.shrink(first)))
        return self.deconv2(self.deconv1(mid)) + bicubic


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)


def main():
    import config
    import numpy as np
    import time

    in_channels = 4
    batch_size = 1
    res = 20

    fsrcnn = FSRCNN(
        maps=4,
        in_channels=in_channels,
        outer_channels=56,
        inner_channels=12,
    ).to('cpu')

    x = torch.randn((batch_size, in_channels, res, res)).to('cpu')

    out = fsrcnn(x)
    print(out.shape)
    t0 = time.time()
    for _ in range(100):
        out = fsrcnn(x)
    print((time.time()-t0)/100)

    from torchsummaryX import summary
    summary(fsrcnn, x)
    params_size = 0.24
    for_back_size = 2.20
    gpu_memory_Mb = 15*1000 # 1750
    batch_size = (gpu_memory_Mb - params_size)/(for_back_size)

    batch_pow_2 = np.uint(2**np.uint(np.log2(batch_size)))
    print(f"On the current gpu the best batch size is: {batch_pow_2}")


if __name__ == '__main__':
    main()
