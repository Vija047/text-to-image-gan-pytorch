import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator

# Hyperparameters
noise_dim = 100
embed_dim = 256
img_channels = 3
lr = 0.0002
batch_size = 64
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
G = Generator(noise_dim, embed_dim, img_channels).to(device)
D = Discriminator(img_channels=img_channels, embed_dim=embed_dim).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Dummy dataset loader (infinite loop)
def get_fake_data_loader():
    while True:
        yield torch.randn(batch_size, img_channels, 64, 64), torch.randn(batch_size, embed_dim)

dataloader = get_fake_data_loader()

# Training loop
for epoch in range(epochs):
    for i in range(100):  # 100 batches per epoch
        real_imgs, captions = next(dataloader)
        real_imgs, captions = real_imgs.to(device), captions.to(device)

        valid = torch.ones(batch_size, device=device)  # ✅ Fixed shape
        fake = torch.zeros(batch_size, device=device)  # ✅ Fixed shape

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        z = torch.randn(batch_size, noise_dim, device=device)
        with torch.no_grad():
            gen_imgs = G(z, captions)

        real_loss = criterion(D(real_imgs, captions), valid)
        fake_loss = criterion(D(gen_imgs, captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        gen_imgs = G(z, captions)
        g_loss = criterion(D(gen_imgs, captions), valid)
        g_loss.backward()
        optimizer_G.step()

        # Logging
        if i % 20 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/100] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# Save the trained generator
torch.save(G.state_dict(), "generator.pth")
print("✅ Generator model saved as 'generator.pth'")
