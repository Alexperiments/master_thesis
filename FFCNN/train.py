import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
import numpy as np
from tqdm import tqdm

import config
from model import FFCNN, initialize_weights
from dataset import MyImageFolder
from utils import load_checkpoint, save_checkpoint, plot_examples


def wandb_init():
    wandb.init(
        entity='aled',
        project="Tesi-ML-FFCNN",
        config={
            "Architecture": "FFCNN",
            "Learning Rate": config.LEARNING_RATE,
            "Batch Size": config.BATCH_SIZE,
            "Max Epochs": config.NUM_EPOCHS,
        },
    )


def train_fn(train_loader, val_loader, model, opt, l1, scaler, scheduler):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for low_res, high_res in loop:
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            super_res = model(low_res)
            loss = l1(super_res, high_res)
        wandb.log({"L1 train loss": loss})
        loop.set_postfix(L1=loss.item())
        losses.append(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        model.eval()
        val_loss = []
        for low_res, high_res in val_loader:
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)

            super_res = model(low_res)
            val_loss.append(l1(super_res, high_res).item())
        wandb.log({"L1 val loss": np.mean(val_loss)})
        model.train()

    scheduler.step(sum(losses)/len(losses))


def main():
    dataset = MyImageFolder()
    train_dataset, val_dataset = random_split(dataset, [9216, 784])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0,1152))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0,100))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = FFCNN(maps=10).to(config.DEVICE)
    initialize_weights(model)
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
    l1 = nn.L1Loss()

    model.train()

    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT,
            model,
            opt,
            scheduler,
        )

    wandb_init()
    for epoch in range(1, config.NUM_EPOCHS+1):
        train_fn(train_loader, val_loader, model, opt, l1, scaler, scheduler)
        print("{0}/{1}".format(epoch,config.NUM_EPOCHS))

        if epoch % 100 == 0:
            if config.SAVE_MODEL:
                save_checkpoint(model, opt, scheduler, filename=config.CHECKPOINT)
        if config.SAVE_IMG_CHKPNT:
            if epoch % 100 == 0:
                plot_examples(config.TRAIN_FOLDER, model, 'checkpoints/'+str(epoch)+'/')

    wandb.finish()


if __name__ == "__main__":
    main()
