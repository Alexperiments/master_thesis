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
    def __init__(self, scaling=4, in_channels=1, maps=4, d=30, s=10):
        super().__init__()
        self.extract = ConvBlock(
            in_channels,
            out_channels=d,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=d,
            out_channels=2*s,
            kernel_size=9,
            stride=scaling//2,
            padding=4,
            output_padding=1
        )
        self.act1 = nn.PReLU()
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=2*s,
            out_channels=s,
            kernel_size=9,
            stride=scaling//2,
            padding=4,
            output_padding=1
        )
        self.act2 = nn.PReLU()
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=s,
            out_channels=1,
            kernel_size=9,
            stride=scaling//2,
            padding=4,
            output_padding=1
        )
        self.act3 = nn.PReLU()

    def forward(self, x):
        first = self.extract(x)
        return self.act3(self.deconv3(self.act2(self.deconv2(self.act1(self.deconv1(first))))))


class FFCNN(nn.Module):
    def __init__(self, in_features=2, in_channels=1, maps=10, low_res=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, low_res)
        self.act1 = nn.PReLU()
        self.fc2 = nn.Linear(low_res, low_res*low_res)
        self.act2 = nn.PReLU()
        self.fsrcnn = FSRCNN(maps=maps)
        self.in_c = in_channels
        self.lr = low_res

    def forward(self, x):
        first = self.act1(self.fc1(x.float()))
        second = self.act2(self.fc2(first))
        reshape = second.view(-1, self.in_c, self.lr, self.lr)
        image = self.fsrcnn(reshape)
        return image


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)

def main():
    from torchsummary import summary
    import config
    import numpy as np

    in_features = 2
    batch_size = 64

    ffcnn = FFCNN().to(config.DEVICE)

    x = torch.randn((batch_size, in_features)).to(config.DEVICE)
    out = ffcnn(x)
    print(out.shape)

    summary(ffcnn, (1, 20, 20))
    params_size = 0.24
    for_back_size = 1.48
    gpu_memory_Mb = 1750
    batch_size = (gpu_memory_Mb - params_size)/(for_back_size)
    # round to the
    batch_pow_2 = np.uint(2**np.uint(np.log2(batch_size)))
    print(f"On the current gpu the best batch size is: {batch_pow_2}")

if __name__ == '__main__':
    main()
