"""
https://www.facebook.com/groups/209358916852889/permalink/1046214066500699/
Further prompts
- see through
- strip bikini
- micro bikini
- nipple cover sexy
- sexy, white skin, straight face
- naked
Add multiple prompts
Result: Face is distorted
"""
import itertools
import math
import os.path
import random

import torch.backends.mps
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything, resize_for_condition_image

seed: int = 8811
seed_everything(seed)
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
prompt = "best quality, masterpiece, (photorealistic:1.4), 1girl, light smile, a girl sitting on the beach bed. She crosses her legs and wearing 2 piece string bikini"

negative_prompt: str = "low resolution, blur, bad quality, distort, bad shape, skinny, turn back, bad face, distort face,"
low_res_img = load_image("sasithorn_bikini_beach/s.png")
out_dir: str = "sasithorn_bikini_beach"
print(f"source_image.size: {low_res_img.size}")

import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)
pipeline.enable_attention_slicing()

upscaled_image = pipeline(prompt=prompt, negative_prompt=negative_prompt,
                          image=low_res_img).images[0]
upscaled_image.save(f"{out_dir}/upsampled.png")
