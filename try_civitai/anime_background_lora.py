"""
Part of anime_background experiments.
This file focus on Lora.
Experiment the real input background picture to anime.
https://civitai.com/models/35960/flat-2d-animerge
https://civitai.com/models/13910/thicker-lines-anime-style-lora-mix
https://civitai.com/models/87191/animechar-with-clothings-makai-tenshi-mix-or-sei-tenshi-djibril
"""
import itertools
# Let's load the popular vermeer image
import os
import typing as typ
import random
from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

from set_seed import seed_everything
from read_lora import load_lora_weights_orig

#NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device.
# If you want this op to be added in priority during the prototype phase of this feature,
# please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix,
# you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
# WARNING: this will be slower than running natively on MPS.
seed: int = 1111
based_model: str = "runwayml/stable-diffusion-v1-5"  # Based model
device = "cpu"
image = load_image(
    "./sources/46350.jpg"
)
output_dir: str = "anime_background_lora"

image = np.array(image)
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(resized, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

lora_list: typ.List[str] = [
    "angelMagicalClothes_v3.safetensors",
    "thickline_fp16.safetensors"
]
control_nets: typ.List[str] = [
    "sd-controlnet-canny",
    "sd-controlnet-hed",
    "sd-controlnet-scribble",
]
multipliers: typ.List[float] = list(range(0, 30, 2))
guidances: typ.List[float] = list(range(0, 30, 2))
combined_list = list(itertools.product(lora_list, control_nets, multipliers, guidances))

# Shuffle the combined list
random.shuffle(combined_list)


total_combinations: int = len(combined_list)
print(f"Total combinations are: {total_combinations}")
for item in tqdm(combined_list, total=total_combinations):
    lora, control_model_name, multiplier, guidance_scale = item
    lora_path = f"civitai_loras/{lora}"
    controlnet = ControlNetModel.from_pretrained(f"lllyasviel/{control_model_name}").to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        based_model,
        safety_checker=None,
        controlnet=controlnet,
        low_cpu_mem_usage=False,
    )
    pipe = load_lora_weights_orig(pipe, lora_path, multiplier, device, torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    generator = torch.manual_seed(seed)
    seed_everything(seed)

    prompt: str = "clean, two tables, 4 chairs, glass door, glass wall, dim light"
    negative_prompt: str = "low quality, dirty, damage, dark"
    num_images_per_prompt: int = 4

    # check existing file
    if not os.path.exists(f"{output_dir}/{lora}_{control_model_name}_{guidance_scale}_{multiplier}_0.png"):
        print("==============================")
        print(f"Running: {output_dir}/{lora}_{control_model_name}_{guidance_scale}_{multiplier}.png")
        out_images = pipe(
            prompt, num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=30, generator=generator, image=canny_image,
            guidance_scale=guidance_scale, negative_prompt=negative_prompt
        )
        for idx, image in enumerate(out_images.images):
            filename = f"{output_dir}/{lora}_{control_model_name}_{guidance_scale}_{multiplier}_{idx}.png"
            image.save(filename)
