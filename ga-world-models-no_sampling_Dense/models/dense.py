import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):
    """Image to latent space model

    Use dense layer
    """

    def __init__(self, img_channels, latent_size, m):
        super(Dense, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(img_channels*64*64, 64)
        self.fc2 = nn.Linear(64, latent_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        latent = self.fc2(x)
        return latent