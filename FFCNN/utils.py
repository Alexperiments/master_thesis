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


def plot_examples(low_res_folder, model, target_folder):
    files = os.listdir(low_res_folder)
    os.system(f"mkdir -p {target_folder}")
    model.eval()
    for file in files:
        path_test = os.path.join(low_res_folder, file)
        test_array = np.load(path_test)

        with torch.no_grad():
            upscaled = model(
                config.transform(test_array)
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        upscaled_2d = upscaled.squeeze(0).squeeze(0)
        int_upscaled = np.uint8(upscaled_2d.cpu()*255)
        cv2.imwrite(target_folder + file + ".png", int_upscaled)
    model.train()
