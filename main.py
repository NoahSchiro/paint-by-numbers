import torch
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import time

from data import get_data
from models import Discriminator, Generator

############ HYPER PARAMETERS ######################

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This is the size of the output image
IMAGE_SIZE  = 128
LATENT_SIZE = 256
STATS       = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # TODO: How can we improve these?

# Theres a tweet by Yann LeCun that says your
# batch size likely never needs to be larger than
# 32
BATCH_SIZE = 32
EPOCHS     = 200
LR_D       = 1e-4
LR_G       = 1e-3

static_latent = torch.randn(1, LATENT_SIZE, device=DEVICE)
def save_img(g, dir, epoch):

    def denorm(img_tensors):
            return img_tensors * STATS[1][0] + STATS[0][0]
    g.eval()
    img = denorm(g(static_latent))
    g.train()
    save_image(img, f"{dir}/epoch{epoch}.png")
 

def train(g, d, dl, dir):
    
    torch.cuda.empty_cache()

    # Save discriminator / generator losses
    loss_d = []
    loss_g = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(d.parameters(), lr=LR_D, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(g.parameters(), lr=LR_G, betas=(0.5, 0.999))

    for epoch in range(1, EPOCHS):

        start = time.time()

        for batch, (real_imgs, _) in enumerate(dl):

            real_imgs = real_imgs.to(DEVICE)

            # Clear gradients
            opt_d.zero_grad()

            # Pass real images through discriminator (we are expecting discriminator
            # to say "1" for all these)
            real_preds   = d(real_imgs)
            real_targets = torch.ones(real_imgs.size(0), 1, device=DEVICE)
            real_loss    = F.binary_cross_entropy(real_preds, real_targets)
            
            # Generate fake images
            latent = torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
            fake_images = g(latent)

            # Pass fake images through discriminator (we are expecting discriminator
            # to say "0" for all these
            fake_targets = torch.zeros(fake_images.size(0), 1, device=DEVICE)
            fake_preds = d(fake_images)
            fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)

            # Update discriminator weights. Note this is dependent on the
            # discriminators ability to tell the difference between real and fake
            # so we need to combine these losses
            loss = real_loss + fake_loss
            loss.backward()
            opt_d.step()

            print(f"Epoch: {epoch}; Batch: {batch}/{len(dl)}")
            print(f"D loss: {loss.item():.4f};", end=' ')
            loss_d.append(loss.item())

            # Clear generator gradients
            opt_g.zero_grad()
            
            # Generate fake images with random vector
            latent = torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
            fake_images = g(latent)
            
            # Try to fool the discriminator
            preds   = d(fake_images)
            targets = torch.ones(BATCH_SIZE, 1, device=DEVICE)
            loss    = F.binary_cross_entropy(preds, targets)
            
            # Update generator weights
            loss.backward()
            opt_g.step()
            
            print(f"G loss: {loss.item():.4f}")
            loss_g.append(loss.item())

        # At the end of an epoch, try saving an image
        save_img(g, dir, epoch)

        stop = time.time()
        print(f"Time to complete epoch: {stop - start}s")

    return loss_d, loss_g


def save(dir, g, d, loss_g, loss_d):
    # Save the models out
    torch.save(g.state_dict(), f"{dir}generator.pth")
    torch.save(d.state_dict(), f"{dir}discriminator.pth")
    
if __name__=="__main__":

    data = get_data(IMAGE_SIZE, STATS)

    dl = DataLoader(data, BATCH_SIZE, shuffle=True)

    print("Starting...")
    print(f"Dataset size:    {len(data)}")
    print(f"Dataloader size: {len(dl)}")
    print(f"Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Device:          {DEVICE}")

    # Testing discriminator
    d = Discriminator(IMAGE_SIZE).to(DEVICE)

    # Testing generator
    g = Generator(LATENT_SIZE, IMAGE_SIZE).to(DEVICE)

    save_dir = f"models/img_{IMAGE_SIZE}x{IMAGE_SIZE}_epochs{EPOCHS}/"

    # Train
    loss_d, loss_g = train(g, d, dl, save_dir)

    # Save the model states
    save(save_dir, g, d, loss_d, loss_g)

   
