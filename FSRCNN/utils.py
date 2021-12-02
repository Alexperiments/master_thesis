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
        int_sr = np.uint8(sr.cpu() * 255)
        cv2.imwrite(target_folder + file + ".png", int_sr)
    model.train()


def max_difference_loss(x, y):
    diff = torch.amax(y, dim=(2, 3)) - torch.amax(x, dim=(2, 3))
    return torch.mean(torch.abs(diff))