"""Web interface for user do text2image."""
import os
import random

import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


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
    model_path = "sdxl-sarit"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, safety_checker=None, requires_safety_checker=False)
    pipe.to(device)
    images = pipe(prompt=prompt, num_images_per_prompt=3).images
    # for idx,img in enumerate(images):
    #     img.save(f"{idx}.png")
    return images


def main() -> None:
    """Run the main function."""
    seed_everything(42)
    input_text = gr.Textbox(lines=2, label="Enter a sentence. The processing time takes about 15 seconds.")
    # output_images = gr.Image(label="Generated Image", type="numpy", multiple=True)
    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    ).style(columns=[2], rows=[2], object_fit="contain", height="auto")

    demo = gr.Interface(fn=generate_image_interface, inputs=input_text, outputs=gallery)
    demo.launch(
        # share=True
    )


if __name__ == "__main__":
    main()
