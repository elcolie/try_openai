import math

import torch
from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils import load_image

device: str = "mps" if torch.backends.mps.is_available() else "cpu"
low_res_img = load_image("/Users/sarit/million/IMG_5696.JPG").convert("RGB")
width, height = low_res_img.size
ratio: float = width / height
percentage: float = 0.05
new_width: int = math.floor(percentage * width)
new_height: int = math.floor(percentage * height)
print(f"{low_res_img.size}")
print(f"New size: {new_width, new_height}")
low_res_img = low_res_img.resize((new_width, new_height))

prompt: str = "Naked girl in rope bondage"
negative_prompt: str = "distort, bad quality"
num_images_per_prompt: int = 100

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16"
)
pipeline = pipeline.to(device)

upscaled_image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=low_res_img,
    num_inference_steps=100,
).images[0]
upscaled_image.save("output/rope.png")
