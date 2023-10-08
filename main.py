import torch
from torch.utils.data import DataLoader

from data import get_data
from models import Discriminator, Generator


############ HYPER PARAMETERS ######################

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This is the size of the output image
IMAGE_SIZE = 64
LATENT_SIZE = 128

# Theres a tweet by Yann LeCun that says your
# batch size likely never needs to be larger than
# 32
BATCH_SIZE = 32
EPOCHS     = 100

if __name__=="__main__":

    data = get_data(IMAGE_SIZE)

    dl = DataLoader(data, BATCH_SIZE, shuffle=True)

    print(f"Dataset size:    {len(data)}")
    print(f"Dataloader size: {len(dl)}")
    print(f"Device:          {DEVICE}")

    # Testing discriminator
    d = Discriminator(IMAGE_SIZE).to(DEVICE)

    # Testing generator
    g  = Generator(LATENT_SIZE, IMAGE_SIZE).to(DEVICE)
    ex = torch.randn(1, LATENT_SIZE).to(DEVICE)

    # Generate random image
    gen_out = g.forward(ex)

    # Discriminator tries to see if it's real
    real = d.forward(gen_out)

    print(real)


