import torch
import config
import cv2
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


def calculate_max_min():
    lr_path = os.path.join('train_data', 'lr')
    hr_path = os.path.join('train_data', 'hr')
    files = os.listdir(lr_path)
    lr = []
    hr = []
    for file in files[:9000]:
        lr_file_path = os.path.join(lr_path, file)
        hr_file_path = os.path.join(hr_path, file)

        hr_file = np.float32(np.load(hr_file_path))
        hr.append(torch.from_numpy(hr_file))

        lr_file = np.float32(np.load(lr_file_path))
        lr.append(torch.from_numpy(lr_file))

    lr = torch.stack(lr).to(config.DEVICE)
    hr = torch.stack(hr).to(config.DEVICE)

    lr_max = torch.amax(lr, axis=(0, 2, 3))
    lr_min = torch.amin(lr, axis=(0, 2, 3))

    hr_max = torch.amax(hr, axis=(0, 2, 3))
    hr_min = torch.amin(hr, axis=(0, 2, 3))

    delta_min = hr_min - lr_min
    delta_max = hr_max - lr_max

    print(f"Max deltas {delta_max.cpu().numpy() - config.NORM_MAX}")
    print(f"Min deltas {delta_min.cpu().numpy() - config.NORM_MIN}")