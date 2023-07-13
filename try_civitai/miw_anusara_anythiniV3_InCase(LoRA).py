"""
https://www.facebook.com/groups/209358916852889/permalink/1046931313095641/
Influence by:
https://www.youtube.com/watch?v=BO4GTG1Gb-4&ab_channel=SafiiClon
Dependencies:
https://civitai.com/models/66/anything-v3
https://civitai.com/models/6648/incase-style-lora
User convert script to convert ckpt file, but do not include --safetensors
"""
import itertools
import os.path
import random
import typing as typ

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from tqdm import tqdm

from read_lora import load_lora_weights_orig
from set_seed import seed_everything, resize_for_condition_image

seed: int = 8888
seed_everything(seed)
# NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"
print(f"Device: {device}")
image = resize_for_condition_image(load_image("./sources/miw.jpeg"), 512)
out_dir: str = "anime_miw_anythingV3_LoRA"

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

prompt: str = "masterpiece, best quality, ultra-detailed, illustration, (1girl), bikini, pool, half body"
negative_prompt: str = "low quality, bad hands"
num_images_per_prompt: int = 1
num_inference_steps: int = 100

lora_path: str = "../ai_files/ics_a3_lora.safetensors"
control_nets: typ.List[str] = [
    "control_v11p_sd15_canny",
    "control_v11p_sd15_scribble"
]
guidances: typ.List[float] = [round(0.1 * _, 3) for _ in range(0, 22, 2)]
multipliers: typ.List[float] = [round(0.1 * _, 3) for _ in range(0, 22, 2)]
models: typ.List[str] = [
    "anythingV3_fp16",
    "animePastelDream",
    "flat2DAnimerge",
    "majicmixRealistic_v6"
]
combined_list = list(itertools.product(control_nets, guidances, multipliers, models))

# Shuffle the combined list
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    control_net, guidance_scale, multiplier, model_id = item
    controlnet = ControlNetModel.from_pretrained(
        f"lllyasviel/{control_net}"
    ).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        f"../ai_directory/{model_id}",
        safety_checker=None,
        controlnet=controlnet,
    ).to(device)
    pipe = load_lora_weights_orig(pipe, lora_path, multiplier, device, torch.float32)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    generator = torch.manual_seed(seed)

    filename: str = f"{out_dir}/{model_id}_{control_net}_{guidance_scale}_{multiplier}_0.png"
    if not os.path.exists(filename):
        try:
            out_images = pipe(
                prompt, num_inference_steps=num_inference_steps, generator=generator, image=canny_image,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale, negative_prompt=negative_prompt
            )
            for idx, image in enumerate(out_images.images):
                image.save(f"{out_dir}/{model_id}_{control_net}_{guidance_scale}_{multiplier}_{idx}.png")
        except Exception as err:
            print(err)
            with open(f"{out_dir}/{model_id}_{control_net}_{guidance_scale}_{multiplier}_0.txt", "a") as file:
                file.write("Dead file")
            continue
