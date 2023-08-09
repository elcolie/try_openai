"""Multiple view same human actor experiments."""
import itertools
import os.path
import random
import typing as typ
import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from tqdm import tqdm

from solve_77_limits import get_pipeline_embeds
from read_lora import load_lora_weights
from set_seed import seed_everything

seed: int = 11212
seed_everything(seed)
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"

models = [
    ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
    ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
    ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
    ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
    ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
    ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
]

wood_men: typ.List[typ.Tuple[str, str]] = [
    ("karate", "wood_man/karate.png"),
    ("one_leg_stand", "wood_man/one_leg_stand.png")
]
views: typ.List[str] = [
    "Medium shot",
    # "Long shot",
    # "Two-shot"
]
loras = [
    # https://civitai.com/models/124460/hayakawanagisajpidol
    ("hayakawanagisa", "hayakawanagisa", "../ai_files/loras/hayakawanagisa_lora-06.safetensors"),
    # https://civitai.com/models/124321/twice-nayeon
    ("nayeonlorashy", "nayeonlorashy", "../ai_files/loras/nayeonlorashy.safetensors"),
]
lora_multiplier: float = 2.0
combined_list = list(itertools.product(models, wood_men, loras, views))

# Shuffle the combined list
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    (model_name, model_path), (out_name, source_file), (prompt, lora_name, lora_file), view = item
    filename: str = f"images/{model_name}_{lora_name}_{out_name}_{view}.png"
    openpose_file: str = f"openposes/{out_name}.png"
    try:
        if not os.path.exists(filename):
            if not os.path.exists(openpose_file):
                image = load_image(source_file)
                openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
                image = openpose(image)
                image.save(openpose_file)
            else:
                print(f"Use existing file {openpose_file}")
                image = load_image(openpose_file)

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose"
            ).to(device)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_path, controlnet=controlnet,
                requires_safety_checker=False, safety_checker=None
            )
            pipe = load_lora_weights(pipe, lora_file, lora_multiplier, device)
            pipe = pipe.to(device)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            prompt = f"{prompt}, {out_name}, {view}"
            negative_prompt = "malformed"
            prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)
            image = pipe(prompt_embeds=prompt_embeds,
                         negative_prompt_embeds=negative_prompt_embeds,
                         image=image, num_inference_steps=800,
                         width=512, height=512).images[0]

            image.save(filename)
        else:
            print(f"{filename} exists")
    except RuntimeError as err:
        print(err)
        print(f"{filename} is not support")
