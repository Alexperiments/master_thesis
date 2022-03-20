import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
from loss import bright_loss
import wandb

torch.backends.cudnn.benchmark = True

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    brightness_loss,
    g_scaler,
    d_scaler,
    tb_step,
):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 5 * l1(fake, high_res)
            adversarial_loss = 0.1 * -torch.mean(disc(fake))
            br_loss = 1e-3 * brightness_loss(fake, high_res)
            gen_loss = l1_loss + adversarial_loss + br_loss

        wandb.log(
            {
                "l1_loss": l1_loss, "adversarial_loss": adversarial_loss, 'brightness_loss': br_loss,
                'gradient_penalty':gp, 'loss_critic':loss_critic, 'Gen loss':gen_loss,
            }
        )

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


def main():
    dataset = MyImageFolder(root_dir=config.TRAIN_FOLDER)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    disc = Discriminator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    brightness_loss = bright_loss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    wandb.init(
        entity='aled',
        project="Tesi-ML-ESRGAN",
        config={
        "learning_rate": config.LEARNING_RATE,
        "architecture": "ESRGAN",
        }
    )

    for epoch in range(config.NUM_EPOCHS+1):
        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            brightness_loss,
            g_scaler,
            d_scaler,
            tb_step,
        )

        print("{0}/{1}".format(epoch,config.NUM_EPOCHS))
        if epoch % 100 == 0:
            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        if config.SAVE_IMG_CHKPNT:
            if epoch % 50 == 0:
                plot_examples(config.TRAIN_FOLDER + "lr/", gen, 'checkpoints/'+str(epoch)+'/')

    wandb.finish()


if __name__ == "__main__":
    try_model = False

    if try_model:
        gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        plot_examples(config.TRAIN_FOLDER + "lr/", gen, 'upscaled/')
    else:
        # This will train from scratch
        main()
