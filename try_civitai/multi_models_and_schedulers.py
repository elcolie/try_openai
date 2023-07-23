"""
https://www.facebook.com/groups/209358916852889/permalink/1051779302610842/
"""
import itertools
import math
import os.path
import random

import diffusers
import numpy as np
import torch
# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything

human_name: str = "yui_anyaphan"
out_dir: str = f"{human_name}_bikini_beach/4"
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

init_image = load_image(f"sources/{human_name}/raw/4.jpeg")
width, height = init_image.size
size_factor: float = 0.5
new_width, new_height = math.floor(width * size_factor / 8) * 8, math.floor(height * size_factor / 8) * 8

init_image = init_image.resize((new_width, new_height))

seed: int = 8811
seed_everything(seed)

generator = torch.Generator(device=device).manual_seed(seed)

mask_background_image = load_image(
    f"sources/{human_name}/cropped/mask_4.png"
)
mask_background_image = mask_background_image.resize((new_width, new_height))


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = 1  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


base_prompt = ""
additional_prompts = [
    "A massive tsunami wave crashing down on a coastal town, causing destruction and chaos ,oil paiting, award winning photography, Bokeh, Depth of Field, HDR, bloom, Chromatic Aberration ,Photorealistic,extremely detailed, trending on artstation, trending on CGsociety, Intricate, High Detail, dramatic, art by midjourney"
]
negative_prompt: str = "disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, (watermark)"
strengths = [1, ]
# guidance_scales = [round(0.1 * _, 3) for _ in range(70, 252, 2)]
guidance_scales = [10, 15, 20, 30, 40]
# eta_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4, 6, 8, 10]
eta_list = [1]

models = [
    ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
    ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
    ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
    ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
    ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
]
schedulers = [
    ("LMSDiscreteScheduler", diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler),
    ("DDIMScheduler", diffusers.schedulers.scheduling_ddim.DDIMScheduler),
    ("DPMSolverMultistepScheduler", diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler),
    ("EulerDiscreteScheduler", diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler),
    ("PNDMScheduler", diffusers.schedulers.scheduling_pndm.PNDMScheduler),
    ("DDPMScheduler", diffusers.schedulers.scheduling_ddpm.DDPMScheduler),
    ("EulerAncestralDiscreteScheduler",
     diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler)
]

combined_list = list(itertools.product(
    models, schedulers, strengths, guidance_scales, eta_list, additional_prompts)
)

# Shuffle the combined list
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    (model_name, model_dir), (scheduler_name, scheduler), strength, guidance_scale, eta, prompt = item
    controlnets = [
        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
        ).to(device),
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_dir,
        controlnet=controlnets[0],
        requires_safety_checker=False,
        safety_checker=None
    ).to(device)
    pipe.requires_safety_check = False
    pipe.safety_checker = None
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)

    my_images = [
        make_inpaint_condition(init_image, mask_background_image),
    ]

    filename: str = f"{out_dir}/{model_name}_{scheduler_name}_{strength}_{guidance_scale}_{eta}_{base_prompt} {prompt[:20]}.png"
    print(f"{filename} is running")
    if not os.path.exists(filename):
        # generate image
        image = pipe(
            f"{base_prompt} {prompt}",
            image=init_image,
            mask_image=mask_background_image,
            control_image=my_images[0],
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            generator=generator,
            eta=eta,
            strength=strength,
            guidance_scale=guidance_scale,
            # controlnet_conditioning_scale=[1.0, 0.8]
        ).images[0].save(filename)
    else:
        print(f"{filename} is exists")
