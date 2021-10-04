import torch
from torch import nn
import time
from torchsummary import summary


class ConvBlock(nn.Module):
    '''Same in-out convolution2d-activation block'''
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        padding = (k_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels, k_size, padding=padding)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class CA(nn.Module):
    '''Channel attention'''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        L2_norm = torch.linalg.vector_norm(x, dim=1)
        return torch.div(x, L2_norm.unsqueeze(1))


class RU(nn.Module):
    '''Adaptive weight Residual Unit'''
    def __init__(self, in_channels, lambda_x, lambda_res):
        super().__init__()
        self.lambda_x = lambda_x
        self.lambda_res =lambda_res
        self.reduction = ConvBlock(in_channels, in_channels//2, kernel_size=1)
        self.expansion = ConvBlock(in_channels//2, in_channels, kernel_size=1)

    def forward(self, x):
        residual = self.expansion(self.reduction(x))
        return self.lambda_x*x + self.lambda_res*residual


class ARFB(nn.Module):
    '''Adaptive residual feature block'''
    def __init__(self, in_channels, ):
        super().__init__()
        self.lambda_x = nn.Parameter(torch.tensor(1.))
        self.lambda_res = nn.Parameter(torch.tensor(1.))

        self.ru1 = RU(in_channels, self.lambda_x, self.lambda_res)
        self.ru2 = RU(in_channels, self.lambda_x, self.lambda_res)
        self.conv1 = ConvBlock(in_channels*2, in_channels*2, kernel_size=1)
        self.conv3 = ConvBlock(in_channels*2, in_channels, kernel_size=3)

    def forward(self, x):
        first_ru = self.ru1(x)
        second_ru = self.ru2(first_ru)
        cat = torch.cat([first_ru, second_ru], dim=1)
        cat = self.conv1(cat)
        cat = self.conv3(cat)
        return self.lambda_x*x + self.lambda_res*cat


class HFM(nn.Module):
    '''High-frequency filtering module'''
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        out = self.upsample(self.avgpool(x))
        return x - out


class EMHA(nn.Module):
    '''Efficient Multi-head attention'''
    '''
    The variables names are as similar to the original paper as possible:
    s: splitting factor before the ScaledDotProduct attention
    m: number of "head" in the Multi-Head module
    embed_size: in the paper it is defined as C1 = C*k^2 where C is the number
        of channels after the unfolding (or embedding), k is the kernel size used
        during the unfolding
    B: batch size
    N: "number of patches" equals to WxH
    O_i: ScaledDotProduct output for the i-th segment
    '''
    def __init__(self, embed_size, splitting_factor=4, num_heads=8):
        super().__init__()
        self.embed_size = embed_size
        self.m = num_heads
        self.s = splitting_factor
        # The number of channel will be reduced by 2 by the reduction module
        self.head_dim = embed_size//(2*num_heads)

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.reduction = nn.Conv1d(embed_size, embed_size//2, kernel_size=1)
        self.expansion = nn.Conv1d(embed_size//2, embed_size, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        B = x.shape[0]
        N = x.shape[2]

        # Prepare the input sequence for the m heads and project it to Q,K,V
        x = x.reshape(B, N, self.m, self.head_dim)
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # Split the Q,K,V in s segments to reduce the computational expense
        queries = queries.reshape(B, N//self.s, self.m, self.s, self.head_dim)
        keys = keys.reshape(B, N//self.s, self.m, self.s, self.head_dim)
        values = values.reshape(B, N//self.s, self.m, self.s, self.head_dim)

        # ScaledDotProduct using Einstein Summation
        energy = torch.einsum("bimse,bjmse->bmsij", [queries, keys])
        # I choose dim=4, dim=3 would be also possible
        attention = torch.softmax(energy/(self.embed_size ** (1/2) ), dim=4)
        O_i = torch.einsum("bmsni,bimsc ->bnmsc", [attention, values])
        # Concat the s segments
        O = O_i.reshape(B, N, self.m, self.head_dim)
        # Concat the m heads
        out = O.reshape(B, self.embed_size//2, N)
        return self.expansion(out)


class HPB(nn.Module):
    '''High-preserving block'''
    def __init__(self, in_channels):
        super().__init__()
        self.arfb1 = ARFB(in_channels)
        self.hfm = HFM()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.preserving_arfb = ARFB(in_channels)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        self.arfb2 = ARFB(in_channels)
        self.conv1 = ConvBlock(in_channels*2, in_channels, kernel_size=1)
        self.ca = CA()
        self.arfb3 = ARFB(in_channels)

    def forward(self, x):
        high_path = self.hfm(self.arfb1(x))
        low_path = self.downsample(high_path)
        for _ in range(5): low_path = self.preserving_arfb(low_path)
        low_path = self.upsample(low_path)
        high_path = self.arfb2(high_path)
        cat = torch.cat([low_path, high_path], dim=1)
        cat = self.arfb3(self.ca(self.conv1(cat)))
        return cat + x


class MLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        self.mlp(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        return x


class ET(nn.Module):
    '''Efficient transformer'''
    def __init__(self, N, embed_size):
        super().__init__()
        self.layernorm = nn.LayerNorm(N)
        self.linear = nn.Linear(N, N)
        self.emha = EMHA(embed_size)
        self.mlp = MLP(embed_size)

    def forward(self, x):
        first = self.emha(self.layernorm(x)) + x
        second = first + self.mlp(self.layernorm(first))
        return second


class LCB(nn.Module):
    '''Lightweight CNN backbone'''
    def __init__(self, in_channels):
        super().__init__()
        self.preservation = nn.Sequential(
            *[HPB(in_channels) for _ in range(3)]
        )

    def forward(self, x):
        return self.preservation(x)


class LTB(nn.Module):
    '''Lightweight transformer backbone'''
    def __init__(self, num_channel, res=20, k_size=3):
        super().__init__()
        self.embed_size = num_channel*k_size**2
        self.N = res*res

        self.unfold = nn.Unfold(kernel_size=k_size, padding=1)
        self.et = ET(self.N, self.embed_size)
        self.fold = nn.Fold(output_size=(res, res), kernel_size=k_size, padding=1)

    def forward(self, x):
        patches = self.unfold(x)
        return self.fold(self.et(patches))


class ETSR(nn.Module):
    '''Efficient transformer super-resolution'''
    def __init__(self, in_channels=1, out_channels=32, low_res=20, scale=4):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3)
        self.lcb = LCB(out_channels)
        self.ltb = LTB(out_channels, low_res)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.shuffle = nn.PixelShuffle(upscale_factor=scale)
        self.conv3 = ConvBlock(out_channels//(scale*scale), in_channels, kernel_size=3)
        self.conv4 = ConvBlock(out_channels//(scale*scale), in_channels, kernel_size=3)

    def forward(self, x):
        initial = self.conv1(x)
        main_way = self.lcb(initial)
        main_way = self.ltb(main_way)
        main_way = self.conv3(self.shuffle(self.conv2(main_way)))
        sub_way = self.conv4(self.shuffle(initial))
        return main_way + sub_way


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)


def test():
    in_channels = 1
    batch = 64
    res = 20

    etsr = ETSR(in_channels)
    #summary(etsr, (in_channels,res,res))
    #Forward/backward pass size (MB): 25.68
    #Params size (MB): 3.29
    #Estimated best batch size for GTX 750 Ti -> ~64
    #Estimated best batch size for TESLA K80 -> ~128-256

    x = torch.randn((batch, in_channels, res, res))

    t0 = time.time()
    out = etsr(x)
    t1 = time.time()
    print(t1-t0)
    print(out.shape)

if __name__ == "__main__":
    test()
