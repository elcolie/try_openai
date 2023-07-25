"""
Demo txt2img to bell.
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/realisticVisionV50_v50VAE.safetensors --from_safetensors --dump_path ai_directory/realisticVisionV50_v50VAE
"""
import itertools
import math
import os.path
import random

import diffusers
import numpy as np
import torch
# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, StableDiffusionPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything
from read_lora import load_lora_weights_orig
from solve_77_limits import get_pipeline_embeds

human_name: str = "bell"
out_dir: str = f"{human_name}"
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
# device: str = "cpu"  # With this LoRA it can't run with mps. It raises RuntimeError: Invalid buffer size: 58.07 GB
print(device)

seed: int = 8811
seed_everything(seed)

base_prompt = "a white girl in pink bikini at the beach"
additional_prompts = [
    "(masterpiece:1.2), best quality,PIXIV",
]
negative_prompt: str = "(low quality, worst quality:1.4),"
strengths = [1, ]
# guidance_scales = [round(0.1 * _, 3) for _ in range(70, 252, 2)]
guidance_scales = [30, ]
# eta_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4, 6, 8, 10]
eta_list = [1]

models = [
    ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
    ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
    ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
    ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
    ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
    ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
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
loras = [
    (None, None),
    ("virginie_efira_v01", "../ai_files/loras/virginie_efira_v01.safetensors"),
    ("Night_scene_20230715120543", "../ai_files/loras/Night_scene_20230715120543.safetensors"),
    ("Oldsaigon-v1.0", "../ai_files/loras/Oldsaigon-v1.0.safetensors"),
]
lora_multipliers = [0.5, 1.0, 1.5, 2.0]

combined_list = list(itertools.product(
    models, schedulers, loras, lora_multipliers, strengths, guidance_scales, eta_list, additional_prompts)
)

# Shuffle the combined list
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    (model_name, model_dir), (scheduler_name, scheduler), (lora_name, lora_file), lora_multiplier, strength, guidance_scale, eta, add_prompt = item
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        requires_safety_checker=False,
        safety_checker=None
    )
    pipe.requires_safety_check = False
    pipe.safety_checker = None
    if lora_name != None:
        # TypeError: Trying to convert BFloat16 to the MPS backend, but it does not have support for that dtype.
        device: str = "cpu"
        print(f"{lora_name} : {device}")
        pipe = pipe.to(device)
        pipe = load_lora_weights_orig(pipe, lora_file, lora_multiplier, device, torch.float32)
    else:
        pipe = pipe.to(device)
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator(device=device).manual_seed(seed)
    prompt: str = f"{base_prompt} {add_prompt}"
    prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)

    filename: str = f"{out_dir}/{model_name}_{scheduler_name}_{lora_name}_{lora_multiplier}_{strength}_{guidance_scale}_{eta}_{base_prompt}_{add_prompt[:20]}.png"
    print(f"{filename} is running")
    if not os.path.exists(filename):
        # generate image
        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=500,
            generator=generator,
            eta=eta,
            guidance_scale=guidance_scale,
        ).images[0].save(filename)
    else:
        print(f"{filename} is exists")
