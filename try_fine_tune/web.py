"""Web interface for user do text2image."""
import os
import random

import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionPipeline


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        pass


def generate_image_interface(prompt: str) -> object:
    """Generate image from prompt."""
    model_path = "sd-pokemon-model"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, requires_safety_checker=False)
    pipe.to(device)
    image = pipe(prompt=prompt).images[0]
    return image


def main() -> None:
    """Run the main function."""
    seed_everything(42)
    input_text = gr.Textbox(lines=2, label="Enter a sentence. The processing time takes about 15 seconds.")
    output_image = gr.Image(label="Generated Image")

    demo = gr.Interface(fn=generate_image_interface, inputs=input_text, outputs=output_image)
    demo.launch(
        # share=True
    )


if __name__ == "__main__":
    main()
