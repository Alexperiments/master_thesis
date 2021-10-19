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


class FFCNN(nn.Module):
    def __init__(self, in_features=4, in_channels=1, low_res=20):
        super().__init__()
        self.fc1 = nn.Linear(in_features, low_res)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(low_res, low_res*low_res)
        self.act2 = nn.ReLU()
        self.fsrcnn = FSRCNN()
        self.in_c = in_channels
        self.lr = low_res

    def forward(self, x):
        initial = self.act2(self.fc2(self.act1(self.fc1(x.float()))))
        reshape = initial.view(-1, self.in_c, self.lr, self.lr)
        image = self.fsrcnn(reshape)
        return image


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)

def main():
    in_features = 4
    batch_size = 64

    ffcnn = FFCNN()

    x = torch.randn((batch_size, in_features))
    out = ffcnn(x)
    print(out.shape)

if __name__ == '__main__':
    main()
