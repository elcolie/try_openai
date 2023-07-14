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
num_images_per_prompt: int = 1
num_inference_steps: int = 50
strengths = [8, 9, 10]
guidance_scales = [8, 9, 10, 11, 12]
eta_list = list(range(4, 11))
calc_size: int = 512
base_prompt = "4k, ultra resolution, sexy, white skin, straight face, sit cross legged, crowded beach, blue sky"
additional_prompts = ["micro bikini", "strip bikini", "micro bikini", "nipple cover", "naked"]
combined_list = list(itertools.product(strengths, guidance_scales, eta_list, additional_prompts))

# Shuffle the combined list
random.shuffle(combined_list)

model_id: str = "runwayml/stable-diffusion-inpainting"
negative_prompt: str = "low resolution, blur, bad quality, distort, bad shape, skinny, turn back, bad face, distort face,"
source_image = load_image("sources/sasithorn.jpeg")
masked_image = load_image("sources/masked_sasithorn.png")
out_dir: str = "sasithorn_bikini_beach"
file_fix: str = "sasithorn_inpaint_"
print(f"source_image.size: {source_image.size}")
size_factor: float = 0.7
width, height = source_image.size

for item in tqdm(combined_list, total=len(combined_list)):
    strength, guidance_scale, eta, prompt = item
    strength = 0.1 * strength
    eta = 0.1 * eta
    filename: str = f"{out_dir}/{file_fix}_{strength}_{guidance_scale}_{eta}_{prompt}_0.png"
    try:
        if not os.path.exists(filename):
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                safety_checker=None,
            )
            pipe = pipe.to(device)
            # pipe.enable_sequential_cpu_offload()
            result = pipe(
                prompt=f"{base_prompt}, {prompt}",
                negative_prompt=negative_prompt,
                # image=resize_for_condition_image(source_image, calc_size),
                # mask_image=resize_for_condition_image(masked_image, calc_size),
                image=source_image,
                mask_image=masked_image,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                eta=eta,
                width=int(math.floor(width * size_factor / 8) * 8),
                height=int(math.floor(height * size_factor / 8) * 8)
            )
            for idx, image in enumerate(result.images):
                filename: str = f"{out_dir}/{file_fix}_{strength}_{guidance_scale}_{eta}_{prompt}_{idx}.png"
                image.save(filename)
    except Exception as err:
        print(err)
        print(f"{out_dir}/{file_fix}_{strength}_{guidance_scale}_{eta}_{prompt}.png is impossible")
        continue
