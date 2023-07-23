"""
https://www.facebook.com/photo?fbid=1345782422948415&set=gm.1051340332654739&idorvanity=209358916852889
To convert model from .safetensors to directory
# python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/henmixrealV10_henmixrealV10.safetensors --from_safetensors --dump_path ai_directory/henmixrealV10_henmixrealV10
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
from read_lora import load_lora_weights_orig

human_name: str = "noname"
out_dir: str = f"{human_name}"
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"  # With this LoRA it can't run with mps. It raises RuntimeError: Invalid buffer size: 58.07 GB
print(device)

init_image = load_image(f"sources/{human_name}/noname.jpeg")
width, height = init_image.size
size_factor: float = 0.8
new_width, new_height = math.floor(width * size_factor / 8) * 8, math.floor(height * size_factor / 8) * 8

init_image = init_image.resize((new_width, new_height))

seed: int = 8811
seed_everything(seed)

generator = torch.Generator(device=device).manual_seed(seed)

mask_background_image = load_image(
    f"sources/{human_name}/u2net_noname1_mask.png"
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
    "(masterpiece:1.2), best quality,PIXIV,Night scene, <lora:Night scene_20230715120543:1>",
    "Ultra-realistic 8k CG,masterpiece,best quality,(photorealistic:1.2),HDR,RAW photo, (film grain:1.1), ((Depth of field)),(looking at viewer:1.2), street, saigon, 1girl, pants,  (full body)), aodai, <lora:saigon:0.6>, <lora:aodai:0.7>, (masterpiece,best quality:1.5)"
]
negative_prompt: str = "EasyNegative, badhandsv5-neg,Subtitles,word,"
strengths = [1, ]
# guidance_scales = [round(0.1 * _, 3) for _ in range(70, 252, 2)]
guidance_scales = [10, 15, 20, 30, 40]
# eta_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4, 6, 8, 10]
eta_list = [1]

models = [
    ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
    # ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
    # ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
    # ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
    # ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    # ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
]
schedulers = [
    # ("LMSDiscreteScheduler", diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler),
    # ("DDIMScheduler", diffusers.schedulers.scheduling_ddim.DDIMScheduler),
    # ("DPMSolverMultistepScheduler", diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler),
    # ("EulerDiscreteScheduler", diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler),
    # ("PNDMScheduler", diffusers.schedulers.scheduling_pndm.PNDMScheduler),
    # ("DDPMScheduler", diffusers.schedulers.scheduling_ddpm.DDPMScheduler),
    ("EulerAncestralDiscreteScheduler",
     diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler)
]
loras = [
    (None, None),
    ("Night_scene", "../ai_files/loras/Night_scene_20230715120543.safetensors"),
    ("Saigon", "../ai_files/loras/Oldsaigon-v1.0.safetensors"),
]
lora_multipliers = [0.5, 1.0]

combined_list = list(itertools.product(
    models, schedulers, loras, lora_multipliers, strengths, guidance_scales, eta_list, additional_prompts)
)

# Shuffle the combined list
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    (model_name, model_dir), (scheduler_name, scheduler), (lora_name, lora_file), lora_multiplier, strength, guidance_scale, eta, prompt = item
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
    if lora_name != None:
        pipe = load_lora_weights_orig(pipe, lora_file, lora_multiplier, device, torch.float32)
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)

    my_images = [
        make_inpaint_condition(init_image, mask_background_image),
    ]

    filename: str = f"{out_dir}/{model_name}_{scheduler_name}_{lora_name}_{strength}_{guidance_scale}_{eta}_{base_prompt} {prompt[:20]}.png"
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
        ).images[0].save(filename)
    else:
        print(f"{filename} is exists")
