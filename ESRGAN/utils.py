import torch
import os
import config
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import save_image


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen, target_folder):
    files = os.listdir(low_res_folder)
    os.system(f"mkdir -p {target_folder}")
    gen.eval()
    for file in files:
        path_test = os.path.join(config.TEST_FOLDER+"lr/", file)
        test_array = np.load(path_test)

        with torch.no_grad():
            upscaled = gen(
                config.transform(test_array)
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        upscaled_2d = upscaled.squeeze(0).squeeze(0)
        int_upscaled = np.uint8(upscaled_2d.cpu()*255)
        cv2.imwrite(target_folder + file + ".png", int_upscaled)
    gen.train()


def preferred_batch_size(gpu_memory_Mb):
    # Run these to know the size of the parameters and the tensors involved
    #gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    #summary(gen, (1,20,20))

    #disc = Discriminator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    #summary(disc, (1,80,80))

    params_size = 64 + 90
    for_back_size = 162 + 23
    batch_size = (gpu_memory_Mb - params_size)/(for_back_size)
    # round to the
    batch_pow_2 = np.uint(2**np.uint(np.log2(batch_size)))
    print(f"On the current gpu the best batch size is: {batch_pow_2}")
