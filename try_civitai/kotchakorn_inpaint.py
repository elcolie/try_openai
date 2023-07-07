"""
https://www.facebook.com/photo/?fbid=230317606517173&set=gm.1042375796884526&idorvanity=209358916852889
Remove bikini.
"""
import itertools
import os.path
import random

from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything

seed: int = 8888
seed_everything(seed)
device: str = "cpu"
num_images_per_prompt: int = 1
num_inference_steps: int = 100
strengths = list(range(1, 11))
guidance_scales = list(range(0, 11))
eta_list = list(range(0, 11))
# image_guidance_scales = [2]
# guidance_scales = [8, 11]

combined_list = list(itertools.product(strengths, guidance_scales, eta_list))

# Shuffle the combined list
random.shuffle(combined_list)

model_id = "runwayml/stable-diffusion-inpainting"
prompt = "Replace her cloth with sexy bikini, shoulder bag, boob"
negative_prompt: str = "blur, bad quality, beach, bad shape, skinny"
source_image = load_image("sources/koch.jpeg")
masked_image = load_image("sources/masked_koch2.jpg")
out_dir: str = "kotchakorn_inpaint"

for item in tqdm(combined_list):
    strength, guidance_scale, eta = item
    strength = 0.1 * strength
    eta = 0.1 * eta
    filename: str = f"{out_dir}/kot_inpaint_{strength}_{guidance_scale}_{eta}.png"
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
                image=source_image,
                mask_image=masked_image,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                eta=eta,
            )
            for image in result.images:
                filename: str = f"{out_dir}/kot_inpaint_{strength}_{guidance_scale}_{eta}.png"
                image.save(filename)
    except Exception:
        print(f"{out_dir}/kot_inpaint_{strength}_{guidance_scale}_{eta}.png is impossible")
        continue
