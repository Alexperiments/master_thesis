import argparse
import os
import numpy as np

import torch
import config as cfg
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from torch.utils.data import DataLoader
from model import FSRCNN, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
import wandb

import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS


def train_fn(train_loader, val_loader, model, opt, loss, scaler):
    loop = tqdm(train_loader, leave=True)
    loss_all_batches = []
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(cfg.DEVICE)
        low_res = low_res.to(cfg.DEVICE)

        with torch.cuda.amp.autocast():
            super_res = model(low_res)
            train_loss = loss(super_res, high_res)
            loss_all_batches.append(train_loss.item())
        #wandb.log({"train loss": train_loss, "LR": cfg.LEARNING_RATE})
        loop.set_postfix(train_loss=train_loss.item())

        opt.zero_grad()
        scaler.scale(train_loss).backward()
        scaler.step(opt)
        scaler.update()

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


def main(config):
    train_dataset = MyImageFolder(root_dir='/home/ale/Scrivania/Tesi-ML/FSRCNN/'+cfg.TRAIN_FOLDER)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )
    val_dataset = MyImageFolder(root_dir='/home/ale/Scrivania/Tesi-ML/FSRCNN/'+cfg.TEST_FOLDER)
    val_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=cfg.NUM_WORKERS
    )
    model = FSRCNN(in_channels=cfg.IMG_CHANNELS, maps=config['hidden']).to(cfg.DEVICE)
    initialize_weights(model)
    opt = optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
    loss = nn.L1Loss()

    model.train()

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, cfg.NUM_EPOCHS+1):
        train_fn(train_loader, val_loader, model, opt, loss, scaler)
        #print("{0}/{1}".format(epoch,cfg.NUM_EPOCHS))
        if epoch % 100 == 0:
            # This saves the model to the trial directory
            torch.save(model, "./model.pth")


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
            "training_iteration": 100
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1  # set this for GPUs
        },
        num_samples=8,
        config={
            "lr": tune.loguniform(1e-5, 4e-3),
            "hidden": 4,
            "batch_size": tune.choice([32, 64, 128, 256]),
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
