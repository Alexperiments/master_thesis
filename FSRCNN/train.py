import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples, plot_difference
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import FSRCNN, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder, MultiEpochsDataLoader
import wandb
import numpy as np
import time


def wandb_init():
    wandb.init(
        entity='aled',
        project="Tesi-ML-FSRCNN",
        config={
            "Architecture": "FSRCNN",
            "Learning Rate": config.LEARNING_RATE,
            "Batch Size": config.BATCH_SIZE,
            "Max Epochs": config.NUM_EPOCHS,
        }
    )


def train_fn(train_loader, val_loader, model, opt, l1, scaler, scheduler):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            super_res = model(low_res)
            loss = l1(super_res, high_res)
        wandb.log({"L1 train loss": loss, "LR": config.LEARNING_RATE})
        loop.set_postfix(L1=loss.item())
        losses.append(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
        second = time.time()
    '''
    with torch.no_grad():
        model.eval()
        val_loss = []
        for low_res, high_res in val_loader:
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)

            super_res = model(low_res)
            val_loss.append(l1(super_res, high_res).item())
        #wandb.log({"L1 val loss": np.mean(val_loss)})
        model.train()
    '''
    scheduler.step(sum(losses)/len(losses))


def main():
    dataset = MyImageFolder()
    train_dataset, val_dataset = random_split(dataset, [8000, 2000])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0,2048))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0,10))

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    model = FSRCNN(maps=4).to(config.DEVICE)
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
            config.LEARNING_RATE,
        )

    wandb_init()

    for epoch in range(1, config.NUM_EPOCHS+1):
        train_fn(train_loader, val_loader, model, opt, l1, scaler, scheduler)
        print("{0}/{1}".format(epoch,config.NUM_EPOCHS))

        if epoch % 100 == 0:
            if config.SAVE_MODEL:
                save_checkpoint(model, opt, filename=config.CHECKPOINT)
        if config.SAVE_IMG_CHKPNT:
            if epoch % 100 == 0:
                plot_examples(config.TEST_FOLDER + "lr/", model, 'checkpoints/'+str(epoch)+'/')

    wandb.finish()


if __name__ == "__main__":
    try_model = False

    if try_model:
        model = FSRCNN(maps=4).to(config.DEVICE)
        opt = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT,
            model,
            opt,
            config.LEARNING_RATE,
        )
        #plot_examples(config.TEST_FOLDER + "lr/", model, 'upscaled/')
        plot_difference(config.TRAIN_FOLDER, model, 'differences/')

    else:
        main()
