import gradio as gr
import torch
from models.generator import Generator
from utils import embed_text, tensor_to_pil

# Load Generator
noise_dim = 100
embed_dim = 256
img_channels = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(noise_dim, embed_dim, img_channels).to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

# Inference function
def generate_image_gradio(prompt):
    z = torch.randn(1, noise_dim).to(device)
    embed = embed_text(prompt).to(device)
    with torch.no_grad():
        img = G(z, embed)
    return tensor_to_pil(img.squeeze(0))

# Launch app
iface = gr.Interface(fn=generate_image_gradio, inputs="text", outputs="image", title="Text to Image GAN")
iface.launch()
