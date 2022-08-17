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
        group="global_hyperparameters_2"
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

<<<<<<< HEAD

def main(config):
    wandb_init(config)
=======
    scheduler.step(sum(losses)/len(losses))


def main(config):
    dataset = MyImageFolder()
    train_dataset, val_dataset = random_split(dataset, [18944, 1056])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0,128))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0,10))
>>>>>>> main

    dataset = MyImageFolder(working_directory)
    train_dataset, val_dataset = random_split(dataset, [61419, 4096])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, 4096))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 512))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=cfg.NUM_WORKERS,
    )

    model = FSRCNN(
        maps=cfg.MAPS,
        in_channels=cfg.IMG_CHANNELS,
        outer_channels=cfg.OUTER_CHANNELS,
        inner_channels=cfg.INNER_CHANNELS,
    ).to(cfg.DEVICE)
    initialize_weights(model)
    opt = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(config["beta1"], config["beta2"])
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
<<<<<<< HEAD
    ray.init(object_store_memory=1e9)  # , address='auto')

    config_dict = {
        "lr": tune.loguniform(5e-4, 5e-2),
        "batch_size": tune.choice([32, 64]),
        "beta1": tune.uniform(0.8, 1),
        "beta2": tune.uniform(0.9, 1)
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        reduction_factor=2,
        max_t=600
    )

    bohb_search = TuneBOHB()
    bohb_search = tune.suggest.ConcurrencyLimiter(bohb_search, max_concurrent=8)
=======
    ray.init(object_store_memory=2e8)
    #wandb_init()
    # for early stopping
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=cfg.NUM_EPOCHS,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["val_loss", "training_iteration"]
    )

    bohb_search = TuneBOHB(max_concurrent=4)

    config_dict = {
        "lr": tune.grid_search(1e-4, 2e-4, 4e-4, 8e-4, 16e-4, 32e-4, 64e-4, 128e-4),
        "batch_size": tune.grid_search(128, 256, 512, 1024, 2048, 4096),
        # wandb config
        "wandb":{
            "entity": 'aled',
            "project": "Optimize FSRCNN",
            "api_key_file": ".wandbapi.txt",
            "log_config": True
        }
    }
>>>>>>> main

    analysis = tune.run(
        main,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
<<<<<<< HEAD
        resources_per_trial={
            "cpu": 8,
            "gpu": 0.5
        },
        num_samples=128,
        config=config_dict,
        metric="train_loss",
        mode="min",
=======
        progress_reporter=reporter,
        stop={
            "training_iteration": cfg.NUM_EPOCHS
        },
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        num_samples=1,
        config=config_dict,
        loggers = DEFAULT_LOGGERS + (WandbLogger, )
>>>>>>> main
    )

    print("Best config is:", analysis.best_config)
