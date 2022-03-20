import argparse
import os
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from tqdm import tqdm

import config
from new_model import FSRCNN, initialize_weights
from dataset import MyImageFolder
from utils import load_checkpoint, save_checkpoint, plot_examples

import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS


def train_fn(train_loader, val_loader, model, opt, loss, scaler, scheduler):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for low_res, high_res in loop:
        high_res = high_res.to(cfg.DEVICE)
        low_res = low_res.to(cfg.DEVICE)
        with torch.cuda.amp.autocast():
            super_res = model(low_res)
            train_loss = loss(super_res, high_res)
            loss_all_batches.append(train_loss.item())

        loop.set_postfix(train_loss=train_loss.item())
        losses.append(train_loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        model.eval()
        val_loss = []
        for low_res, high_res in val_loader:
            high_res = high_res.to(cfg.DEVICE)
            low_res = low_res.to(cfg.DEVICE)

            super_res = model(low_res)
            val_loss.append(loss(super_res, high_res).item())
        tune.report(val_loss=np.mean(val_loss))
        model.train()

    scheduler.step(sum(losses)/len(losses))


def main(config):
    dataset = MyImageFolder()
    train_dataset, val_dataset = random_split(dataset, [9216, 784])
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0,100))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0,100))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
    )

    model = FSRCNN().to(cfg.DEVICE)
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
        factor=0.5,
        patience=10,
        verbose=True
    )

    model.train()

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, cfg.NUM_EPOCHS+1):
        train_fn(train_loader, val_loader, model, opt, loss, scaler)


if __name__ == "__main__":
    ray.init(object_store_memory=8e7)
    #wandb_init()
    # for early stopping
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1000,
        reduction_factor=4,
        stop_last_trials=False)


    bohb_search = TuneBOHB(
        # space=config_space,  # If you want to set the space manually
        max_concurrent=4)

    analysis = tune.run(
        main,
        metric="val_loss",
        mode="min",
        name="bohb_test",
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        stop={
            "training_iteration": 300
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        num_samples=50,
        config={
            "beta1": tune.uniform(0.8, 1),
            "beta2": tune.uniform(0.9, 1),
            # wandb config
            "wandb":{
                "entity": 'aled',
                "project": "Optimize FSRCNN",
                "api_key_file": ".wandbapi.txt",
                "log_config": True
            }
        },
        loggers = DEFAULT_LOGGERS + (WandbLogger, )
    )
    #wandb.finish()
    print("Best config is:", analysis.best_config)
