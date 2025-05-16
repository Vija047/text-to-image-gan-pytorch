import torch
from torchvision.utils import save_image
from models.generator import Generator
from utils import embed_text
import os

# Parameters (must match training)
noise_dim = 100
embed_dim = 256
img_channels = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model
model_path = "generator.pth"
G = Generator(noise_dim, embed_dim, img_channels).to(device)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found! Please train the generator and save it as '{model_path}'.")
G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()

# Image generation
def generate_image(prompt: str, out_path: str = "output.png"):
    z = torch.randn(1, noise_dim).to(device)
    embed = embed_text(prompt).to(device)
    with torch.no_grad():
        generated_img = G(z, embed)
    save_image(generated_img, out_path)
    print(f"✅ Image saved to {out_path}")

# Run
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    generate_image("a dog playing with a ball", out_path="data/sample.png")
