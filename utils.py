import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Dummy embedder for now — replace with a real NLP encoder later
def embed_text(text, embed_dim=256):
    torch.manual_seed(abs(hash(text)) % (2**32))  # deterministic embeddings
    return F.normalize(torch.randn(1, embed_dim), dim=1)

# Convert tensor to PIL Image
def tensor_to_pil(tensor):
    tensor = (tensor + 1) / 2  # [-1,1] → [0,1]
    return to_pil_image(tensor.cpu())
