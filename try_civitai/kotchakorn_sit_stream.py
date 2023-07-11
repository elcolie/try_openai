"""
https://www.facebook.com/photo/?fbid=230317606517173&set=gm.1042375796884526&idorvanity=209358916852889
Request from Kotchakorn(Nan)
Further prompts
- see through
- strip bikini
- micro bikini
- nipple cover sexy
- naked
"""
import itertools
import math
import os.path
import random

import torch.backends.mps
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything

seed: int = 88888
seed_everything(seed)
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
num_images_per_prompt: int = 1
num_inference_steps: int = 200
strengths = [8, 9, 10, 11, 12]
guidance_scales = [8, 9, 10, 11, 12]
eta_list = list(range(4, 11))
combined_list = list(itertools.product(strengths, guidance_scales, eta_list))

# Shuffle the combined list
random.shuffle(combined_list)

model_id = "runwayml/stable-diffusion-inpainting"
prompt = "see through micro bikini, sexy, white skin, straight face, bent sit forward"
negative_prompt: str = "blur, bad quality, distort, bad shape, skinny, turn back, bad face"
source_image = load_image("sources/kotchakorn_base.jpg")
masked_image = load_image("sources/masked_kotchakorn_base.png")
out_dir: str = "kotchakorn_sit_stream"
print(f"source_image.size: {source_image.size}")
size_factor: float = 0.7
width, height = source_image.size

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


for item in tqdm(combined_list, total=len(combined_list)):
    strength, guidance_scale, eta = item
    strength = 0.1 * strength
    eta = 0.1 * eta
    filename: str = f"{out_dir}/kot_inpaint_{strength}_{guidance_scale}_{eta}_0.png"
    try:
        if not os.path.exists(filename):
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                safety_checker=None,
            )
            pipe = pipe.to(device)
            # pipe.enable_sequential_cpu_offload()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=resize_for_condition_image(source_image, 1024),
                mask_image=resize_for_condition_image(masked_image, 1024),
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                eta=eta,
                width=int(math.floor(width * size_factor / 8) * 8),
                height=int(math.floor(height * size_factor / 8) * 8)
            )
            for idx, image in enumerate(result.images):
                filename: str = f"{out_dir}/kot_inpaint_{strength}_{guidance_scale}_{eta}_{idx}.png"
                image.save(filename)
    except Exception as err:
        print(err)
        print(f"{out_dir}/kot_inpaint_{strength}_{guidance_scale}_{eta}.png is impossible")
        continue
