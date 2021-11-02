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
from model import FSRCNN
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

        minn = lr.min() + config.NORM_MIN
        maxx = lr.max() + config.NORM_MAX

        with torch.no_grad():
            sr = model(
                config.transform(lr, minn, maxx)
                .unsqueeze(0)
                .to(config.DEVICE)
            ).cpu()
        sr = config.reverse_transform(sr, minn, maxx)
        diff = (sr-hr)/hr

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for i, matrix in enumerate([hr, sr, diff]):
            if i == 2:
                minn = -0.01
                maxx = 0.01
            im = axs[i].imshow(matrix, cmap='hot', vmin=minn, vmax=maxx)
            plt.colorbar(im, ax=axs[i])
        axs[0].title.set_text('HR')
        axs[1].title.set_text('SR')
        axs[2].title.set_text(f'(SR-HR)/HR {diff.min():.2f}')
        plt.savefig(f"{target}{file}.png", dpi=300)
        plt.close(fig)
    model.train()


def check_distribution(source, model, target, num_samples=1000):
    os.system(f"mkdir -p {target}")
    lr = []
    hr = []
    lr_path = os.path.join(source, 'lr')
    hr_path = os.path.join(source, 'hr')
    files = os.listdir(lr_path)
    random.shuffle(files)
    for file in files[:num_samples]:
        lr_file_path = os.path.join(lr_path, file)
        hr_file_path = os.path.join(hr_path, file)

        hr_file = np.load(hr_file_path)
        hr.append(torch.from_numpy(hr_file))

        lr_file = np.load(lr_file_path)
        lr.append(torch.from_numpy(lr_file))

    lr = torch.stack(lr).to(config.DEVICE)
    hr = torch.stack(hr).to(config.DEVICE)

    min, _ = torch.min(lr.view(num_samples, 400), dim=1)
    max, _ = torch.max(lr.view(num_samples, 400), dim=1)
    min = min + torch.tensor(config.NORM_MIN)
    max = max + torch.tensor(config.NORM_MAX)
    minn = min.unsqueeze(1).unsqueeze(1).expand(num_samples, 20, 20)
    maxx = max.unsqueeze(1).unsqueeze(1).expand(num_samples, 20, 20)
    delta = torch.sub(maxx, minn)
    sub_lr = torch.sub(lr, minn)
    norm_lr = torch.div(sub_lr, delta)

    norm_lr = norm_lr.unsqueeze(1)

    sr_out = model(norm_lr).squeeze(1)

    minn = min.unsqueeze(1).unsqueeze(1).expand(num_samples, 80, 80)
    maxx = max.unsqueeze(1).unsqueeze(1).expand(num_samples, 80, 80)
    delta = torch.sub(maxx, minn)
    sr_out = torch.mul(sr_out, delta) + minn

    sr_mean = torch.mean(sr_out, dim=[1,2])
    sr_std = torch.std(sr_out, dim=[1,2])

    hr_mean = torch.mean(hr, dim=[1,2])
    hr_std = torch.std(hr, dim=[1,2])

    diff_mean = torch.sub(sr_mean, hr_mean).detach().cpu().numpy()
    diff_std = torch.sub(sr_std, hr_std).detach().cpu().numpy()

    plt.hist(diff_mean, bins=20)
    plt.savefig(f"{target}mean_difference.png", dpi=300)
    plt.close()
    plt.hist(diff_std, bins=20, alpha=0.5)
    plt.savefig(f"{target}std_difference.png", dpi=300)


def check_parameters():
    df = pd.read_csv(config.TRAIN_FOLDER + "parameters.txt", sep='\t')
    df.plot.scatter(' log10Mb ', ' log10Rb ', s=0.5, color='black')
    plt.show()


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

#plot_examples(config.TRAIN_FOLDER + "lr/", model, 'upscaled/')
plot_difference(config.TRAIN_FOLDER, model, 'differences/')
check_distribution(config.TRAIN_FOLDER, model, 'distributions/')
# check_parameters()
