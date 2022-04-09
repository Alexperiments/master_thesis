import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import numpy as np
from tqdm import tqdm
import datetime

import config
from model import FSRCNN, initialize_weights
from dataset import MyImageFolder, MultiEpochsDataLoader, SingleExampleDataFolder
from utils import load_checkpoint, save_checkpoint, plot_examples


def wandb_init(config_dict):
    wandb.init(
        entity='aled',
        project="Tesi-ML-FSRCNN",
        config=config_dict,
        settings=wandb.Settings(start_method='fork'),
        mode="offline",
	group=f"{datetime.datetime.now()}",
    )


def init_multiprocess(rank, world_size):
    import os

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)


def cleanup_multiprocess():
    dist.destroy_process_group()


def run_ddp_main(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


config_dict = {
    "Architecture": "FSRCNN",
    "Learning Rate": config.LEARNING_RATE,
    "Batch Size": config.BATCH_SIZE,
    "Max Epochs": config.NUM_EPOCHS,
    "N. GPUs": config.GPU_NUMBER,
    "N. workers": config.NUM_WORKERS,
    "Img. channels": config.IMG_CHANNELS,
}


def train_fn(train_loader, val_loader, model, opt, l1, scheduler, rank):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for low_res, high_res in loop:
        # low_res = low_res.view(-1, 1, config.LOW_RES, config.LOW_RES)
        # high_res = high_res.view(-1, 1, config.HIGH_RES, config.HIGH_RES)

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
            # low_res = low_res.view(-1, 1, config.LOW_RES, config.LOW_RES)
            # high_res = high_res.view(-1, 1, config.HIGH_RES, config.HIGH_RES)

            super_res = model(low_res)
            val_loss.append(l1(super_res, high_res).item())
        if config.LOG_REPORT:
            wandb.log({"L1 val loss": np.mean(val_loss)})
        model.train()
    if config.LOG_REPORT:
        wandb.log({"L1 train loss": sum(losses) / len(losses)})
    scheduler.step(sum(losses) / len(losses))


def main(rank, world_size):
    dataset = MyImageFolder()
    train_dataset, val_dataset = random_split(dataset, [18944, 1056])

    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, 9472))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 528))
    
    config_dict["Training size"] = len(train_dataset)
    config_dict["Validation size"] = len(val_dataset)

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

    init_multiprocess(rank=rank, world_size=world_size)

    model = FSRCNN(
        maps=config.MAPS,
        in_channels=config.IMG_CHANNELS,
        outer_channels=config.OUTER_CHANNELS,
        inner_channels=config.INNER_CHANNELS,
    ).to(rank)

    config_dict["maps"] = config.MAPS
    config_dict["in_channels"] = config.IMG_CHANNELS
    config_dict["outer_channels"] = config.OUTER_CHANNELS
    config_dict["inner_channels"] = config.INNER_CHANNELS

    ddp_model = DDP(model, device_ids=[rank], output_device=0)

    initialize_weights(ddp_model)
    opt = optim.Adam(
        ddp_model.parameters(),
        lr=config.LEARNING_RATE
    )
    scheduler = ReduceLROnPlateau(
        opt,
        'min',
        factor=config.LR_DECAY_FACTOR,
        patience=config.DECAY_PATIENCE,
        verbose=True
    )

    config_dict["LR decay factor"] = config.LR_DECAY_FACTOR
    config_dict["Weight decay patience"] = config.DECAY_PATIENCE

    l1 = nn.L1Loss()

    ddp_model.train()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT,
            ddp_model,
            opt,
            scheduler,
        )

    if config.LOG_REPORT: wandb_init(config_dict)
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_fn(train_loader, val_loader, ddp_model, opt, l1, scheduler, rank)
        print("{0}/{1}".format(epoch, config.NUM_EPOCHS))

        if epoch % 20 == 0:
            if config.SAVE_MODEL:
                save_checkpoint(ddp_model, opt, scheduler, filename=config.CHECKPOINT)
        if config.SAVE_IMG_CHKPNT:
            if epoch % 100 == 0:
                plot_examples(config.TRAIN_FOLDER, ddp_model, 'checkpoints/' + str(epoch) + '/')

    if config.LOG_REPORT: wandb.finish()
    cleanup_multiprocess()


if __name__ == "__main__":
    run_ddp_main(main, config.GPU_NUMBER)
