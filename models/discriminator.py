import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, embed_dim=256, feature_dim=64):
        super(Discriminator, self).__init__()

        self.img_branch = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1),  # 3 -> 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1),  # 64 -> 128
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1),  # 128 -> 256
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1),  # 256 -> 512
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.embed_branch = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 8 * 4 * 4),  # 256 -> 8192
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim * 16, feature_dim * 8, 3, 1, 1),  # 1024 -> 512
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0),  # 512 -> 1
            nn.Sigmoid()
        )

    def forward(self, image, embed):
        x_img = self.img_branch(image)          # [B, 512, 4, 4]
        x_embed = self.embed_branch(embed)      # [B, 8192]
        x_embed = x_embed.view(embed.size(0), 512, 4, 4)  # [B, 512, 4, 4]

        x = torch.cat([x_img, x_embed], dim=1) # [B, 1024, 4, 4]

        validity = self.classifier(x)           # [B, 1, 1, 1]
        return validity.view(-1)                # [B]
