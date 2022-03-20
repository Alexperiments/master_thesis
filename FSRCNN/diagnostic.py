import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

import config
from model import FSRCNN
from utils import load_checkpoint

import time


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

        x_lr = np.arange(0, 20, 1)
        x_hr = np.arange(0, 20, 0.25)

        f_bic = interp2d(x_lr, x_lr, lr, kind='cubic')
        bicubic = f_bic(x_hr, x_hr)

        minn = lr.min() + config.NORM_MIN
        maxx = lr.max() + config.NORM_MAX

        with torch.no_grad():
            sr = model(
                config.transform(lr, minn, maxx)
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        sr = config.reverse_transform(sr.cpu(), minn, maxx)
        diff_sr = (sr-hr)/hr
        diff_bi = (bicubic-hr)/hr

        fig, axs = plt.subplots(2, 3, figsize=(13, 7))
        fig.tight_layout(h_pad=2)
        axs = axs.flatten()
        for i, matrix in enumerate([hr, bicubic, sr, lr, diff_bi, diff_sr]):
            if (i == 4) | (i == 5):
                minn = -0.05
                maxx = 0.05
            im = axs[i].imshow(matrix, cmap='hot', vmin=minn, vmax=maxx)
            plt.colorbar(im, ax=axs[i])
        axs[0].title.set_text('HR')
        axs[1].title.set_text('bicubic')
        axs[2].title.set_text('SR')
        axs[3].title.set_text('LR')
        axs[4].title.set_text(f'(bi-HR)/HR {diff_bi.min():.2f}')
        axs[5].title.set_text(f'(SR-HR)/HR {diff_sr.min():.2f}')
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


def bench_time(path, model, num_samples):
    files = os.listdir(path)
    data = []
    for file in files[:num_samples]:
        file_path = os.path.join(path, file)
        numpy_array = np.load(file_path)
        data.append(torch.from_numpy(numpy_array))

    data = torch.stack(data).to(config.DEVICE)

    data = data.unsqueeze(1)

    t0 = time.time()
    sr_out = model(data).squeeze(1)
    delta_t = time.time() - t0
    single_img_time = delta_t/num_samples
    print(f"device: \t\t{config.DEVICE}")
    print(f"processed image: \t{num_samples}")
    print(f"total time: \t\t{delta_t:.6f} s")
    print(f"single image mean: \t{single_img_time:.6f} s")


def plot_worst_or_best(source, model, target, num_samples=30):
    os.system(f"mkdir -p {target}")
    lr = []
    hr = []
    slice = 1000
    lr_path = os.path.join(source, 'lr')
    hr_path = os.path.join(source, 'hr')
    files = os.listdir(lr_path)
    random.shuffle(files)
    for file in files[:slice]:
        lr_file_path = os.path.join(lr_path, file)
        hr_file_path = os.path.join(hr_path, file)

        hr_file = np.load(hr_file_path)
        hr.append(torch.from_numpy(hr_file))

        lr_file = np.load(lr_file_path)
        lr.append(torch.from_numpy(lr_file))

    lr = torch.stack(lr).to(config.DEVICE)
    hr = torch.stack(hr).to(config.DEVICE)

    min, _ = torch.min(lr.view(slice, 400), dim=1)
    max, _ = torch.max(lr.view(slice, 400), dim=1)
    min = min + torch.tensor(config.NORM_MIN)
    max = max + torch.tensor(config.NORM_MAX)
    minn = min.unsqueeze(1).unsqueeze(1).expand(slice, 20, 20)
    maxx = max.unsqueeze(1).unsqueeze(1).expand(slice, 20, 20)
    delta = torch.sub(maxx, minn)
    sub_lr = torch.sub(lr, minn)
    norm_lr = torch.div(sub_lr, delta)

    norm_lr = norm_lr.unsqueeze(1)

    sr = model(norm_lr).squeeze(1)

    minn = min.unsqueeze(1).unsqueeze(1).expand(slice, 80, 80)
    maxx = max.unsqueeze(1).unsqueeze(1).expand(slice, 80, 80)
    delta = torch.sub(maxx, minn)
    sr = torch.mul(sr, delta) + minn

    diff = torch.sub(sr, hr)
    losses = torch.mean(diff, dim=[1,2])
    diff = torch.div(diff, hr)

    sort = abs(losses.cpu().detach().numpy()).argsort()

    hr = hr[sort].cpu()
    sr = sr[sort].cpu()
    diff = diff[sort].cpu()
    minns = minn[sort].cpu()
    maxxs = maxx[sort].cpu()

    for j in range(1, num_samples):
        one_hr = hr[-j, :, :].detach().numpy()
        one_sr = sr[-j, :, :].detach().numpy()
        one_diff = diff[-j, :, :].detach().numpy()
        minn = minns[-j, 0, 0].numpy()
        maxx = maxxs[-j, 0, 0].numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for i, matrix in enumerate([one_hr, one_sr, one_diff]):
            if i == 2:
                minn = -0.05
                maxx = 0.05
            im = axs[i].imshow(matrix, cmap='hot', vmin=minn, vmax=maxx)
            plt.colorbar(im, ax=axs[i])
        axs[0].title.set_text('HR')
        axs[1].title.set_text('SR')
        axs[2].title.set_text(f'(SR-HR)/HR {one_diff.min():.2f}')
        plt.savefig(f"{target}{j}.png", dpi=300)
        plt.close(fig)
    model.train()


def check_parameters():
    df = pd.read_csv(config.TRAIN_FOLDER + "parameters.txt", sep='\t')
    df.plot.scatter(' log10Mb ', ' log10Rb ', s=0.5, color='black')
    plt.show()


def plot_difference_4ch(source, model, target, num_samples=30):
    lr_folder = os.path.join(source, 'lr')
    hr_folder = os.path.join(source, 'hr')
    files = os.listdir(lr_folder)
    random.shuffle(files)
    os.system(f"mkdir -p {target}")
    model.eval()
    for file in files[:num_samples]:
        path_lr = os.path.join(lr_folder, file)
        lr = np.float32(np.load(path_lr))

        path_hr = os.path.join(hr_folder, file)
        hr = np.float32(np.load(path_hr))

        delta_max = config.NORM_MAX.copy()
        delta_min = config.NORM_MIN.copy()

        delta_min = np.swapaxes(np.array([[delta_min]]), 0, 2)
        delta_max = np.swapaxes(np.array([[delta_max]]), 0, 2)
        maxx = np.float32(np.amax(lr, axis=(1, 2), keepdims=True) + delta_max)
        minn = np.float32(np.amin(lr, axis=(1, 2), keepdims=True) + delta_min)

        lr_input = config.transform(lr, minn, maxx)

        lr_input = lr_input.view(-1, 1, config.LOW_RES, config.LOW_RES)

        with torch.no_grad():
            sr = model(
                lr_input
                .to(config.DEVICE)
            ).cpu()
        sr = sr.view(-1, config.HIGH_RES, config.HIGH_RES)
        sr = config.reverse_transform(sr, minn, maxx)
        diff_sr = (sr-hr)/hr

        fig, axs = plt.subplots(4, 4, figsize=(12, 8))
        fig.tight_layout(h_pad=2)
        for i, matrix in enumerate([lr, hr, sr, diff_sr]):
            for j in range(4):
                if i == 3:im = axs[j, i].imshow(matrix[j], vmin=-0.05, vmax=0.05)
                else: im = axs[j, i].imshow(matrix[j])
                plt.colorbar(im, ax=axs[j, i])
        axs[0, 0].title.set_text('LR')
        axs[0, 1].title.set_text('HR')
        axs[0, 2].title.set_text('SR')
        axs[0, 3].title.set_text(f'(SR-HR)/HR')
        axs[0, 0].set_ylabel('density')
        axs[1, 0].set_ylabel('vel. profile')
        axs[2, 0].set_ylabel('vel. disp')
        axs[3, 0].set_ylabel('M/L')
        plt.savefig(f"{target}{file}.png", dpi=300)
        plt.close(fig)
    model.train()


model = FSRCNN(
    maps=5,
    in_channels=1,
    outer_channels=56,
    inner_channels=12,
).to(config.DEVICE)
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
#plot_difference(config.TRAIN_FOLDER, model, 'differences/')
plot_difference_4ch(config.TRAIN_FOLDER, model, 'diagnostic/')
#check_distribution(config.TRAIN_FOLDER, model, 'distributions/', num_samples=10)
#plot_worst_or_best(config.TRAIN_FOLDER, model, 'best_worse/')
#bench_time(config.TRAIN_FOLDER + 'hr/', model, num_samples=100)
# check_parameters()
