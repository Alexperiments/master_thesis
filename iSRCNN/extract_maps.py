import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import numpy as np
import matplotlib.pyplot as plt

import config
from model import FSRCNN
from dataset import MyImageFolder
from utils import load_checkpoint, plot_examples


model = FSRCNN(maps=10).to(config.DEVICE)
opt = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
)
scheduler = ReduceLROnPlateau(
    opt,
    'min',
    factor=0.5,
    patience=10,
    verbose=True
)
load_checkpoint(
    config.CHECKPOINT,
    model,
    opt,
    scheduler
)
model.eval()

test_image = np.load("train_data/lr/3549.npy")
min = test_image.min()
max = test_image.max()
test_image = config.transform(test_image, min, max)
test_image = test_image.to(config.DEVICE).unsqueeze(0)

out = model(test_image).detach().cpu().numpy().squeeze(0)

fig, axs = plt.subplots(3, 4, figsize=(24, 12), sharex=True, sharey=True)

for idx, ax in enumerate(axs.flatten()):
    ax.imshow(out[idx, :, :], cmap='hot')

plt.tight_layout()
plt.show()
