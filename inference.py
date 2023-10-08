import torch
from torchvision.utils import save_image

from models import Generator

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FILE_PATH = "./models/img_64x64_e100/test1generator.pth"
LATENT_SIZE = 128
IMAGE_SIZE  = 64
STATS       = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def save_img(g, id):

    # Create a random vector
    static_latent = torch.randn(1, LATENT_SIZE, device=DEVICE)

    def denorm(img_tensors):
            return img_tensors * STATS[1][0] + STATS[0][0]

    # Create a painting, denormalize
    img = denorm(g(static_latent))

    # Save the image out
    save_image(img, f"models/inference{id}.png")

if __name__=="__main__":
    model = Generator(LATENT_SIZE, IMAGE_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(FILE_PATH, map_location=DEVICE))

    # Generate 10 images
    for i in range(1, 10):
        save_img(model, i)
