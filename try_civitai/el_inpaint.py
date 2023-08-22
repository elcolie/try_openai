"""Remove man from background"""
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

seed: int = 8888862
seed_everything(seed)
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
num_images_per_prompt: int = 1
num_inference_steps: int = 150
models = [
    ("stable-diffusion-inpainting", "runwayml/stable-diffusion-inpainting"),
    # ("abased_v10", "../ai_directory/abased_v10"),
    # ("animeArtDiffusionXL_alpha3", "../ai_directory/animeArtDiffusionXL_alpha3"),  # Can't use with controlnet
    # ("animePastelDream", "../ai_directory/animePastelDream"),
    # ("anythingV3_fp16", "../ai_directory/anythingV3_fp16"),
    # ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    # ("exquisiteDetails_v10", "../ai_directory/exquisiteDetails_v10"),
    # ("flat2DAnimerge", "../ai_directory/flat2DAnimerge"),
    # ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
    # ("kawaiiRealisticAnime_a03", "../ai_directory/kawaiiRealisticAnime_a03"),
    # ("lyricalVivian_v10", "../ai_directory/lyricalVivian_v10"),
    # ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),
    # ("MeinaV10", "../ai_directory/MeinaV10"),
    # ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),
    # ("realdosmix", "../ai_directory/realdosmix"),
    # ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    # ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
    # ("sd_xl_base_0.9", "../ai_directory/sd_xl_base_0.9"), # Can't use with controlnet
]
prompts = [
    "Remove man in the background."
]

combined_list = list(itertools.product(models, prompts))

# Shuffle the combined list
random.shuffle(combined_list)
negative_prompt: str = "blur, bad quality, bad shape, skinny, cloths, limb"
source_image = load_image("/Users/sarit/Desktop/o.jpg")
masked_image = load_image("/Users/sarit/Desktop/o.png")
out_dir: str = "el_inpaint"
print(f"source_image.size: {source_image.size}")
size_factor: float = 0.8
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

# combined_list = list(itertools.product(prompts, strengths, guidance_scales, eta_list))
for item in tqdm(combined_list, total=len(combined_list)):
    (model_name, model_path), prompt = item
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename: str = f"{out_dir}/{model_name}_{prompt}"
    chk_filename: str = f"{filename}_0.png"
    try:
        if not os.path.exists(chk_filename):
            print(device)
            print(chk_filename)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(device)
            # pipe.enable_sequential_cpu_offload()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=resize_for_condition_image(source_image, 1024),
                mask_image=resize_for_condition_image(masked_image, 1024),
                num_inference_steps=num_inference_steps,
                width=int(math.floor(width * size_factor / 8) * 8),
                height=int(math.floor(height * size_factor / 8) * 8),
                num_images_per_prompt=num_images_per_prompt,
            )
            for idx, image in enumerate(result.images):
                image.save(f"{filename}_{idx}.png")
        else:
            print(f"{chk_filename} exists")
    except Exception as err:
        print(err)
        print(f"{filename} is impossible")
        continue
