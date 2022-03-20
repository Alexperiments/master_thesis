import torch
from model import FSRCNN as FSRCNN
import config
from fvcore.nn import FlopCountAnalysis

in_channels = 1
batch_size = 1
res = 20
dataset_size = 10000
epochs = 500
baseline_epoch_time = 8  # seconds
total_dataset_size = 1.5e6
gpu750ti_flops = 1.389e12
gpuv100_flops = 14.13e12
num_gpu = 4
augmentation_factor = 10
maps = 4

v100_flops = num_gpu * gpuv100_flops
flops_ratio = v100_flops / gpu750ti_flops
dataset_ratio = total_dataset_size / dataset_size

expected_time = baseline_epoch_time * epochs * dataset_ratio \
                * augmentation_factor * maps / flops_ratio
real_hours = expected_time / 3600
core_hours = 32 * 2 * real_hours

print(f"Training with augmentation (real hours): {real_hours}")
print(f"Training with augmentation (core hours): {core_hours}")

'''model = FSRCNN(
    maps=4,
    in_channels=in_channels,
    outer_channels=56,
    inner_channels=12,
).to(config.DEVICE)

x = torch.randn((batch_size, in_channels, res, res)).to(config.DEVICE)

flops = FlopCountAnalysis(model, x)
print(flops.total())
print(flops.by_module())'''
