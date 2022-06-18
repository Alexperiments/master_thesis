import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
import numpy as np
from tqdm import tqdm

import config as cfg
from model import FSRCNN, initialize_weights
from dataset import MyImageFolder

import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

working_directory = os.getcwd()


def wandb_init(config_dict={}):
    wandb.init(
        entity='aled',
        project='Optimize FSRCNN',
        config=config_dict,
        settings=wandb.Settings(start_method='fork'),
        mode="offline",
        group="local_hyperparameters"
    )


def train_fn(train_loader, val_loader, model, opt, loss, scheduler):
    loop = tqdm(train_loader, leave=True)
    train_losses = []
    for low_res, high_res in loop:
        high_res = high_res.to(cfg.DEVICE)
        low_res = low_res.to(cfg.DEVICE)
        super_res = model(low_res)
        train_loss = loss(super_res, high_res)
        loop.set_postfix(L1=train_loss.item())
        train_losses.append(train_loss)
        train_loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        model.eval()
        val_losses = []
        for low_res, high_res in val_loader:
            high_res = high_res.to(cfg.DEVICE)
            low_res = low_res.to(cfg.DEVICE)

            super_res = model(low_res)
            val_losses.append(loss(super_res, high_res).item())
        model.train()
        val_loss = sum(val_losses) / len(val_losses)
        train_loss = sum(train_losses) / len(train_losses)
        train_loss = train_loss.cpu().numpy()
    scheduler.step(train_loss)

    return train_loss, val_loss


def main(config):
    wandb_init(config)

    dataset = MyImageFolder(working_directory)
    train_dataset, val_dataset = random_split(dataset, [61440, 4096])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, 4096))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 10))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE  # config["batch_size"],
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE  # config["batch_size"],
        num_workers=cfg.NUM_WORKERS,
    )

    model = FSRCNN(
        maps=config['maps'],  # cfg.MAPS,
        in_channels=cfg.IMG_CHANNELS,
        outer_channels=config['out_channels']  # cfg.OUTER_CHANNELS,
        inner_channels=config['in_channels']  # cfg.INNER_CHANNELS,
    ).to(cfg.DEVICE)
    initialize_weights(model)
    opt = optim.Adam(
        model.parameters(),
        lr=cfg.LAERNING_RATE  # config['lr'],
        # betas=(config["beta1"], config["beta2"])
    )
    loss = nn.L1Loss()
    scheduler = ReduceLROnPlateau(
        opt,
        'min',
        factor=cfg.LR_DECAY_FACTOR,
        patience=cfg.DECAY_PATIENCE,
        verbose=True
    )

    model.train()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"Epoch: {epoch}/{cfg.NUM_EPOCHS}")
        train_loss, val_loss = train_fn(train_loader, val_loader, model, opt, loss, scheduler)

        train_loss = float(train_loss)
        tune.report(train_loss=train_loss)
        if cfg.LOG_REPORT:
                wandb.log({"train_loss": train_loss})
                wandb.log({"val_loss": val_loss})

if __name__ == "__main__":
    ray.init(object_store_memory=1e9)  # , address='auto')

    config_dict = {
        "maps": tune.randint(1,10),
        "out_channels": tune.randint(40, 60),
        "in_channels": tune.randint(6, 16),
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        reduction_factor=2,
        max_t=300
    )

    bohb_search = TuneBOHB()
    bohb_search = tune.suggest.ConcurrencyLimiter(bohb_search, max_concurrent=8)

    analysis = tune.run(
        main,
        name="bohb_test",
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        resources_per_trial={
            "cpu": 6,
            "gpu": 0.5
        },
        num_samples=100,
        config=config_dict,
        metric="train_loss",
        mode="min",
    )

    print("Best config is:", analysis.best_config)
