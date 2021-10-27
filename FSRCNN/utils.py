import torch
import config
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def save_checkpoint(model, optimizer, scheduler, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, scheduler):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])


def plot_examples(source, model, target_folder):
    lr_folder = os.path.join(source, 'lr')
    hr_folder = os.path.join(source, 'hr')
    files = os.listdir(lr_folder)
    os.system(f"mkdir -p {target_folder}")
    model.eval()
    for file in files[:20]:
        lr_path = os.path.join(lr_folder, file)
        hr_path = os.path.join(hr_folder, file)
        lr = np.load(lr_path)
        hr = np.load(hr_path)

        minn = hr.min()
        maxx = hr.max()

        with torch.no_grad():
            sr = model(
                config.transform(lr, minn, maxx)
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        sr = config.reverse_transform(sr, minn, maxx)
        int_sr = np.uint8(sr.cpu()*255)
        cv2.imwrite(target_folder + file + ".png", int_sr)
    model.train()


def plot_difference(source, model, target):
    lr_folder = os.path.join(source, 'lr')
    hr_folder = os.path.join(source, 'hr')
    files = os.listdir(lr_folder)
    os.system(f"mkdir -p {target}")
    model.eval()
    for file in files[:100]:
        path_lr = os.path.join(lr_folder, file)
        lr = np.load(path_lr)

        path_hr = os.path.join(hr_folder, file)
        hr = np.load(path_hr)
        minn = hr.min()
        maxx = hr.max()

        with torch.no_grad():
            sr = model(
                config.transform(lr, minn, maxx)
                .unsqueeze(0)
                .to(config.DEVICE)
            ).cpu()
        sr = config.reverse_transform(sr, minn, maxx)
        diff = sr-hr

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for i, matrix in enumerate([hr, sr, diff]):
            if i == 2:
                minn = min(diff.flatten())
                maxx = max(diff.flatten())
            im = axs[i].imshow(matrix, cmap='hot', vmin=minn, vmax=maxx)
            plt.colorbar(im, ax=axs[i])
        plt.savefig(f"{target}{file}.png", dpi=300)
        plt.close(fig)
    model.train()
