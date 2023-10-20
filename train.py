import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import time
from datetime import timedelta
import os
import logging

from src.data import get_data
from src.models import Discriminator, Generator

############ HYPER PARAMETERS ######################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
SCALER_D = torch.cuda.amp.GradScaler()
SCALER_G = torch.cuda.amp.GradScaler()

# This is the size of the output image
IMAGE_SIZE = 64
LATENT_SIZE = 32
STATS = (0.5229942, 0.48899996, 0.41180329), (0.25899375, 0.24669976, 0.25502672)

BATCH_SIZE = 2 # Generally does not need to be > 32
EPOCHS = 500
LR_D = 1e-4
LR_G = 1e-3

# Where we will save our model to
save_dir = f"models/img_{IMAGE_SIZE}x{IMAGE_SIZE}_epochs{EPOCHS}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set up a logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(f"{save_dir}log.txt"), logging.StreamHandler()],
)

# We set up a random vector to see how the model progresses over time
static_latent = torch.randn(1, LATENT_SIZE, device=DEVICE)


def save_img(g, epoch):
    def denorm(img_tensors):
        return img_tensors * STATS[1][0] + STATS[0][0]

    g.eval()
    with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
        img = denorm(g(static_latent))
    g.train()
    save_image(img, f"{save_dir}/epoch{epoch}.png")


def train(g, d, dl):
    torch.cuda.empty_cache()

    # Create optimizers
    opt_d = torch.optim.Adam(d.parameters(), lr=LR_D)
    opt_g = torch.optim.Adam(g.parameters(), lr=LR_G)

    global_start = time.time()

    for epoch in range(1, EPOCHS+1):
        start = time.time()

        logging.info(f"EPOCH {epoch}")

        # Accumulate loss
        loss_d_acc = 0.0
        loss_g_acc = 0.0

        for batch, (real_imgs, _) in enumerate(dl):
            real_imgs = real_imgs.to(DEVICE)

            # Clear gradients
            opt_d.zero_grad()

            # Pass real images through discriminator (we are expecting discriminator
            # to say "1" for all these)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):

                real_preds = d(real_imgs)
                real_targets = torch.ones(real_imgs.size(0), 1, device=DEVICE)
                real_loss = F.binary_cross_entropy_with_logits(real_preds, real_targets)

                # Generate fake images
                latent = torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
                fake_images = g(latent)

                # Pass fake images through discriminator (we are expecting discriminator
                # to say "0" for all these)
                fake_targets = torch.zeros(fake_images.size(0), 1, device=DEVICE)
                fake_preds = d(fake_images)
                fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)

                # Update discriminator weights. Note this is dependent on the
                # discriminators ability to tell the difference between real and fake
                # so we need to combine these losses
                loss = real_loss + fake_loss

            # Update the discriminator weights (with the grad scaler for mixed precision)
            SCALER_D.scale(loss).backward()
            SCALER_D.step(opt_d)
            SCALER_D.update()

            # Accumulate a loss
            loss_d_acc += loss.item()

            # Clear generator gradients
            opt_g.zero_grad()

            # Generate fake images with random vector
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                latent = torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
                fake_images = g(latent)

                # Try to fool the discriminator
                preds = d(fake_images)
                targets = torch.ones(BATCH_SIZE, 1, device=DEVICE)
                loss = F.binary_cross_entropy_with_logits(preds, targets)

            # Update generator weights (with the grad scaler, this is mixed precision stuff)
            SCALER_G.scale(loss).backward()
            SCALER_G.step(opt_g)
            SCALER_G.update()

            # Accumulate a loss
            loss_g_acc += loss.item()

            # Do some reporting
            if batch % 10 == 0:
                logging.info(f"Batch {batch}/{len(dl)}\n")

                # Compute loss average
                loss_g = loss_g_acc / 10
                loss_d = loss_d_acc / 10

                # Reset the accumulation for the next round
                loss_g_acc = 0.0
                loss_d_acc = 0.0
                logging.info(f"Avg Gen Loss: {loss_g:.3f}")
                logging.info(f"Avg Dis Loss: {loss_d:.3f}\n")

        if epoch % 20 == 0:
            # Save an image
            save_img(g, epoch)
            # Save the model states
            torch.save(g.state_dict(), f"{save_dir}generator{epoch}.pth")
            torch.save(d.state_dict(), f"{save_dir}discriminator{epoch}.pth")

        stop = time.time()
        time_since = timedelta(seconds=(stop - global_start))
        epoch_time = timedelta(seconds=(stop - start))
        remaining = epoch_time * (EPOCHS - epoch)
        logging.info(f"Running time: {str(time_since)}")
        logging.info(f"Epoch time:   {str(epoch_time)}")
        logging.info(f"ETA:          {str(remaining)}")


if __name__ == "__main__":
    data = get_data(IMAGE_SIZE, STATS)

    # I have 8 cpu cores so 8 workers I guess. May
    # need to decrease this if memory is an issue
    dl = DataLoader(data, BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)

    logging.info(f"Dataset size:    {len(data)}")
    logging.info(f"Dataloader size: {len(dl)}")
    logging.info(f"Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
    logging.info(f"Latent size:     {LATENT_SIZE}")
    logging.info(f"Batch size:      {BATCH_SIZE}")
    logging.info(f"Epochs:          {EPOCHS}")
    logging.info(f"Device:          {DEVICE}")

    # Testing discriminator
    d = Discriminator(IMAGE_SIZE).to(DEVICE)

    # Testing generator
    g = Generator(LATENT_SIZE, IMAGE_SIZE).to(DEVICE)

    # Train
    train(g, d, dl)

    # Save the model states
    torch.save(g.state_dict(), f"{save_dir}generator_final.pth")
    torch.save(d.state_dict(), f"{save_dir}discriminator_final.pth")
