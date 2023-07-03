"""
Super resolution experiment.
https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages
"""
import math

import torch
from diffusers import LDMSuperResolutionPipeline
from diffusers.utils import load_image

from set_seed import seed_everything

seed_everything(200)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# let's download an  image
low_res_img = load_image("/Users/sarit/million/IMG_5696.JPG").convert("RGB")
factor: float = 0.06
width, height = low_res_img.size
new_width: int = math.floor(factor * width)
new_height: int = math.floor(factor * height)
low_res_img = low_res_img.resize((new_width, new_height))

# run pipeline in inference (sample random noise and denoise)
upscaled_image = pipeline(
    low_res_img, num_inference_steps=100, eta=1,
).images[0]
# save image
upscaled_image.save("output/ldm_generated_image.png")
