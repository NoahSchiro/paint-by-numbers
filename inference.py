import torch
from torchvision.utils import save_image

from models import Generator

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FILE_PATH = "./models/img_128x128_epochs200/generator.pth"
LATENT_SIZE = 248
IMAGE_SIZE  = 512
STATS       = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

@torch.no_grad()
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

    torch.cuda.empty_cache()

    with torch.no_grad():
        model = Generator(LATENT_SIZE, IMAGE_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(FILE_PATH, map_location=DEVICE))

        model.eval()
        print("Model loaded")

        # Generate 10 images
        for i in range(1, 15+1):
            print(f"Generating image {i}")
            save_img(model, i)
