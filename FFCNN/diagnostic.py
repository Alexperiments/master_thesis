import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from model import FFCNN
from dataset import MyImageFolder
from utils import load_checkpoint, plot_examples


def plot_difference(source, model, target, num_samples=30):
    lr_folder = os.path.join(source, 'lr')
    hr_folder = os.path.join(source, 'hr')
    files = os.listdir(lr_folder)
    random.shuffle(files)
    os.system(f"mkdir -p {target}")
    model.eval()
    for file in files[:num_samples]:
        path_lr = os.path.join(lr_folder, file)
        lr = np.load(path_lr)

        path_hr = os.path.join(hr_folder, file)
        hr = np.load(path_hr)

        with torch.no_grad():
            sr = model(
                config.transform(lr)
                .unsqueeze(0)
                .to(config.DEVICE)
            ).cpu()
        sr = config.reverse_transform(sr)
        diff = (sr-hr)/hr

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for i, matrix in enumerate([hr, sr, diff]):
            im = axs[i].imshow(matrix, cmap='hot')
            plt.colorbar(im, ax=axs[i])
        axs[0].title.set_text('HR')
        axs[1].title.set_text('SR')
        axs[2].title.set_text(f'(SR-HR)/HR {diff.min():.2f}')
        plt.savefig(f"{target}{file}.png", dpi=300)
        plt.close(fig)
    model.train()


model = FFCNN(maps=10).to(config.DEVICE)
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

plot_difference(config.TRAIN_FOLDER, model, 'differences/')
