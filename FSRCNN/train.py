import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import FSRCNN, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
import wandb
import numpy as np


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
        '''opt.zero_grad()
        scaler.scale(l1_loss).backward()
        scaler.step(opt)
        scaler.update()'''

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
    train_dataset = MyImageFolder(root_dir=config.TRAIN_FOLDER)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_dataset = MyImageFolder(root_dir=config.TEST_FOLDER)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    model = FSRCNN(maps=10).to(config.DEVICE)
    initialize_weights(model)
    opt = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        #betas=(0.9714, 0.9897)
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
        model = FSRCNN(maps=10).to(config.DEVICE)
        opt = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            #betas=(0.9714, 0.9897)
        )
        load_checkpoint(
            config.CHECKPOINT,
            model,
            opt,
            config.LEARNING_RATE,
        )
        plot_examples(config.TEST_FOLDER + "lr/", model, 'upscaled/')

    else:
        main()
