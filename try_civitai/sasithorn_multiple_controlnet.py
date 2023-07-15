"""
https://www.facebook.com/groups/209358916852889/permalink/1046214066500699/
Use multiple controlNet
1. Cloth
2. Background
Face distortion will be cut/paste from the original picture
"""
import itertools
import random

import cv2
# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from tqdm import tqdm

from set_seed import seed_everything
from PIL import Image

out_dir: str = "sasithorn_bikini_beach"
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

init_image = load_image("sources/sasithorn.jpeg")
# init_image = init_image.resize((512, 512))

seed: int = 8811
seed_everything(seed)

generator = torch.Generator(device=device).manual_seed(seed)

mask_cloth_image = load_image(
    "sources/masked_cloth_sasithorn.png"
)
mask_background_image = load_image(
    "sources/masked_background_sasithorn.png"
)

def return_array_image(image):
    canny_image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)

    # zero out middle columns of image where pose will be overlayed
    zero_start = canny_image.shape[1] // 4
    zero_end = zero_start + canny_image.shape[1] // 2
    canny_image[:, zero_start:zero_end] = 0

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    return canny_image

# mask_image = mask_image.resize((512, 512))

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


base_prompt = "4k, ultra resolution, sexy, white skin, straight face, sit cross legged, blue sky"
additional_prompts = ["swimsuit", "bikini"]
negative_prompt: str = "low resolution, blur, bad quality, distortion, bad shape, skinny, turn back, bad face, distorted face, ugly face, people, limbs"
strengths = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
guidance_scales = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10]
eta_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4, 6, 8, 10]
combined_list = list(itertools.product(
    strengths, guidance_scales, eta_list, additional_prompts)
)

# Shuffle the combined list
random.shuffle(combined_list)

controlnets = [
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
    ).to(device),
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
    ).to(device),
]
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnets,
    requires_safety_checker=False,
    safety_checker=None
).to(device)
pipe.requires_safety_check = False
pipe.safety_checker = None
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

my_images = [
    make_inpaint_condition(init_image, mask_cloth_image),
    make_inpaint_condition(init_image, mask_background_image)
]

for item in tqdm(combined_list, total=len(combined_list)):
    strength, guidance_scale, eta, add_prompt = item
    # generate image
    image = pipe(
        f"{base_prompt}, {add_prompt}",
        image=my_images,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        generator=generator,
        eta=eta,
        strength=strength,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=[1.0, 0.8]
    ).images[0].save(f"{out_dir}/{strength}_{guidance_scale}_{eta}_{add_prompt}.png")
