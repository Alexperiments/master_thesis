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
    def __init__(self, scaling=4, in_channels=1, maps=4):
        super().__init__()
        self.extract = ConvBlock(
            in_channels,
            out_channels=56,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.shrink = ConvBlock(
            in_channels=56,
            out_channels=12,
            kernel_size=1
        )
        self.map = nn.Sequential(
            *[ConvBlock(
                in_channels=12,
                out_channels=12,
                kernel_size=3,
                padding=1,
                stride=1
            ) for _ in range(maps)]
        )
        self.expand = ConvBlock(
            in_channels=12,
            out_channels=56,
            kernel_size=1
        )
        '''self.deconv = ConvBlock(
            in_channels=56,
            out_channels=1,
            kernel_size=9,
            stride=scaling,
            padding=34
        )'''
        self.deconv = nn.ConvTranspose2d(
            in_channels=56,
            out_channels=1,
            kernel_size=9,
            stride=scaling,
            padding=3,
            output_padding=1
        )

    def forward(self, x):
        first = self.extract(x)
        mid = self.expand(self.map(self.shrink(first)))
        return self.deconv(mid)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)

def main():
    from torchsummary import summary
    import config
    import numpy as np

    in_channels = 1
    batch_size = 64
    res = 20

    fsrcnn = FSRCNN(maps=10).to(config.DEVICE)

    x = torch.randn((batch_size, in_channels, res, res)).to(config.DEVICE)
    out = fsrcnn(x)
    print(out.shape)

    summary(fsrcnn, (1, 20, 20))
    params_size = 0.08
    for_back_size = 2.28
    gpu_memory_Mb = 1750
    batch_size = (gpu_memory_Mb - params_size)/(for_back_size)
    # round to the
    batch_pow_2 = np.uint(2**np.uint(np.log2(batch_size)))
    print(f"On the current gpu the best batch size is: {batch_pow_2}")


if __name__ == '__main__':
    main()
