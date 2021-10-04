import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from torch.utils.data import DataLoader
from model import ETSR, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
import wandb


def wandb_init():
    wandb.init(
        entity='aled',
        project="Tesi-ML-ETSR",
        config={
            "Architecture": "ETSR",
            "Learning Rate": config.LEARNING_RATE,
            "Batch Size": config.BATCH_SIZE,
            "Max Epochs": config.NUM_EPOCHS,
        }
    )


def decay_lr(epoch, epochs_step=200):
    if epoch % epochs_step == 0:
        config.LEARNING_RATE /= 2


def train_fn(loader, etsr, opt, l1, scaler):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            super_res = etsr(low_res)
            l1_loss = l1(super_res, high_res)

        wandb.log({"L1 loss": l1_loss, "LR": config.LEARNING_RATE})
        loop.set_postfix(L1=l1_loss.item())

        opt.zero_grad()
        scaler.scale(l1_loss).backward()
        scaler.step(opt)
        scaler.update()


def main():
    dataset = MyImageFolder(root_dir=config.TRAIN_FOLDER)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    etsr = ETSR(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    initialize_weights(etsr)
    opt = optim.Adam(etsr.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.9))
    l1 = nn.L1Loss()

    etsr.train()

    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            etsr,
            opt,
            config.LEARNING_RATE,
        )

    wandb_init()

    for epoch in range(1, config.NUM_EPOCHS+1):
        train_fn(loader, etsr, opt, l1, scaler)
        print("{0}/{1}".format(epoch,config.NUM_EPOCHS))

        if epoch % 1000 == 0:
            if config.SAVE_MODEL:
                save_checkpoint(etsr, opt, filename=config.CHECKPOINT_GEN)
        if config.SAVE_IMG_CHKPNT:
            if epoch % 1000 == 0:
                plot_examples(config.TRAIN_FOLDER + "lr/", etsr, 'checkpoints/'+str(epoch)+'/')
        if config.LR_DECAY:
            decay_lr(epoch)

    wandb.finish()


if __name__ == "__main__":
    try_model = False

    if try_model:
        etsr = ETSR(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
        opt = optim.Adam(etsr.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.9))
        plot_examples(config.TRAIN_FOLDER + "lr/", etsr, 'upscaled/')

    else:
        main()
