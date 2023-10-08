import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, size):
        super().__init__()

        # in: 3 x size x size
        self.c1 = nn.Conv2d(3, size, kernel_size=4, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(size)
        # out: size x size/2 x size/2

        self.c2 = nn.Conv2d(size, size*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(size*2)
        # out: size*2 x size/4 x size/4

        self.c3 = nn.Conv2d(size*2, size*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(size*4)
        # out: size*4 x size/8 x size/8

        self.c4 = nn.Conv2d(size*4, size*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(size*8)
        # out: size*8 x size/16 x size/16

        self.linear = nn.Linear((size*8) * (size // 16)**2, 1)

        # Activation functions
        self.lReLU   = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.sig     = nn.Sigmoid()


    def forward(self, img):
        img = self.c1(img)
        img = self.b1(img)
        img = self.lReLU(img)

        img = self.c2(img)
        img = self.b2(img)
        img = self.lReLU(img)

        img = self.c3(img)
        img = self.b3(img)
        img = self.lReLU(img)

        img = self.c4(img)
        img = self.b4(img)
        img = self.lReLU(img)

        # Flatten out
        img = self.flatten(img)

        # Pass through linear layer
        img = self.linear(img)

        # Sigmoid activation
        img = self.sig(img)

        return img

class Generator(nn.Module):
    def __init__(self, latent_vec_size, img_size):
        super().__init__()
        self.latent_size = latent_vec_size
        self.img_size = img_size

        # We are doing this so that the tensor sizes line up nice for the output
        self.linear = nn.Linear(self.latent_size, 512*(self.img_size//16)**2)

        # Turn a random vector into an image
        self.ct1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.b1  = nn.BatchNorm2d(256)

        self.ct2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.b2  = nn.BatchNorm2d(128)

        self.ct3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.b3  = nn.BatchNorm2d(64)

        # We need 3 output channels for a 3 color image
        self.cFinal = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)

        # Activations
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, rand_vec):
        assert self.latent_size == rand_vec.shape[1]

        # Again the whole purpose of this is to make our output tensor the right shape
        rand_vec = self.linear(rand_vec)
        rand_vec = self.relu(rand_vec)
        rand_vec = rand_vec.view(rand_vec.size(0), 512, self.img_size // 16, self.img_size // 16)

        img = self.ct1(rand_vec)
        img = self.b1(img)
        img = self.relu(img)

        img = self.ct2(img)
        img = self.b2(img)
        img = self.relu(img)

        img = self.ct3(img)
        img = self.b3(img)
        img = self.relu(img)

        img = self.cFinal(img)
        return self.tanh(img)

