import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
import numpy as np
from tqdm import tqdm

import config
from model import FSRCNN, initialize_weights
from dataset import MyImageFolder, MultiEpochsDataLoader, SingleExampleDataFolder
from utils import load_checkpoint, save_checkpoint, plot_examples

import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.integration.wandb import WandbLogger, wandb_mixin
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import CLIReporter


def train_fn(train_loader, val_loader, model, opt, loss, scaler, scheduler, rank):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for low_res, high_res in loop:
        high_res = high_res.to(rank)
        low_res = low_res.to(rank)
        super_res = model(low_res)
        loss = l1(super_res, high_res)
        loop.set_postfix(L1=loss.item())
        losses.append(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        model.eval()
        val_loss = []
        for low_res, high_res in val_loader:
            high_res = high_res.to(rank)
            low_res = low_res.to(rank)

            super_res = model(low_res)
            val_loss.append(l1(super_res, high_res).item())
        model.train()

    scheduler.step(sum(losses)/len(losses))


def main(config):
    dataset = MyImageFolder()
    train_dataset, val_dataset = random_split(dataset, [18944, 1056])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0,128))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0,10))

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = FSRCNN(
        maps=config.MAPS,
        in_channels=config.IMG_CHANNELS,
        outer_channels=config.OUTER_CHANNELS,
        inner_channels=config.INNER_CHANNELS,
    ).to(cfg.DEVICE)
    initialize_weights(model)
    opt = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=(config['beta1'], config['beta2'])
    )
    loss = nn.L1Loss()
    scheduler = ReduceLROnPlateau(
        opt,
        'min',
        factor=config.LR_DECAY_FACTOR,
        patience=config.DECAY_PATIENCE,
        verbose=True
    )

    model.train()

    for epoch in range(1, cfg.NUM_EPOCHS+1):
        train_fn(train_loader, val_loader, model, opt, loss, scaler)


if __name__ == "__main__":
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

    analysis = tune.run(
        main,
        metric="val_loss",
        mode="min",
        name="bohb_test",
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
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
    )
    #wandb.finish()
    print("Best config is:", analysis.best_config)
