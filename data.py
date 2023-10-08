from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms import functional as F


# I need to create a custom transform
class ResizeAspectRatio:
    def __init__(self, targetWidth, targetHeight):
        self.size = (targetWidth, targetHeight) 

    def __call__(self, img):

        w, h = img.size

        aspect_ratio = float(h) / float(w)

        # If aspect ratio is greater than 1, this means height
        # is larger than width (i.e. a portrait). So the width
        # is the lower bound
        if aspect_ratio > 1:

            # Resize to this
            img = F.resize(img, self.size[0])
        else:
            # Resize to this
            img = F.resize(img, self.size[1])

        # Crop down from the top left corner
        return F.crop(img, 0, 0, self.size[1], self.size[0])
 
def get_data(size):

    targetw, targeth = size, size 
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # TODO: How can we improve these?

    transform = T.Compose([
        ResizeAspectRatio(targetw, targeth),
        T.ToTensor(),
        T.Normalize(*stats)
    ])

    return ImageFolder("./data", transform)
