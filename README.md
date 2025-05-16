# Text-to-Image GAN

A PyTorch implementation of a Generative Adversarial Network (GAN) that creates images from text descriptions.

## Project Structure

- [train.py](train.py) - Training script for the GAN
- [generate.py](generate.py) - Script for generating images from text prompts
- [app.py](app.py) - Gradio web interface for interactive image generation
- [models/](models/)
  - [generator.py](models/generator.py) - Generator network architecture
  - [discriminator.py](models/discriminator.py) - Discriminator network architecture
- [utils.py](utils.py) - Utility functions for text embedding and image processing

## Setup

1. Install dependencies:
```bash
pip install torch torchvision gradio pillow
```

2. Train the model:
```bash
python train.py
```

3. Generate images using either:

Command line:
```bash
python generate.py
```

Web interface:
```bash
python app.py
```

## Model Architecture

The model consists of:
- Generator: Takes noise vector (100d) and text embedding (256d) as input to generate 64x64 RGB images
- Discriminator: Evaluates authenticity of generated images considering both image and text embedding
- Text Embedder: Currently uses a simple random embedding (placeholder for future NLP model)

## Usage

### Command Line Generation

```python
from generate import generate_image

generate_image("a dog playing with a ball", out_path="output.png")
```

### Web Interface

Run `app.py` to launch a Gradio web interface where you can enter text prompts and see generated images in real-time.

## Training Details

- Batch size: 64
- Learning rate: 0.0002
- Epochs: 50
- Optimizer: Adam (β1=0.5, β2=0.999)
- Loss: Binary Cross Entropy

## Outputs

Generated images are saved in:
- Command line: `data/` directory
- Web interface: `.gradio/flagged/` directory

## Future Improvements

- Replace dummy text embedder with proper NLP model
- Improve image resolution
- Add conditional batch normalization
- Implement attention mechanisms
