import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, img_channels, feature_dim=64):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, feature_dim * 8 * 4 * 4),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, embed):
        x = torch.cat([noise, embed], 1)
        x = self.fc(x).view(x.size(0), -1, 4, 4)
        return self.deconv(x)
