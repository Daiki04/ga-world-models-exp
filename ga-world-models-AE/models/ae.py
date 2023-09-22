
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_Encoder(nn.Module):
    """Convolutional decoder for AE
    
    simple AE encoder single layer
    """
    def __init__(self, img_channels, latent_size, m):
        super(AE_Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, stride=2, padding=1) # (3, 64, 64) -> (16, 32, 32)
        # self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=6, stride=4, padding=2) # (3, 64, 64) -> (16, 16, 16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # (16, 16, 16) -> (32, 8, 8)
        self.fc_z = nn.Linear(m, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        z = self.fc_z(x)

        return z
    
class AE_Decoder(nn.Module):
    """Convolutional decoder for AE
    
    simple AE decoder single layer
    """
    def __init__(self, img_channels, latent_size, m):
        super(AE_Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, m)
        self.deconv1 = nn.ConvTranspose2d(m, 16, kernel_size=3, stride=2, padding=1, output_padding=1) # (16, 32, 32) -> (3, 64, 64)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        reconstruction = torch.sigmoid(self.deconv1(x)) #was F.s
        return reconstruction
    

class AE(nn.Module):
    """ AE encoder """
    def __init__(self, img_channels, latent_size, m):
        super(AE, self).__init__()
        self.encoder = AE_Encoder(img_channels, latent_size, m)
        self.decoder = AE_Decoder(img_channels, latent_size, m)

    def forward(self, x): # pylint: disable=arguments-differ
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z