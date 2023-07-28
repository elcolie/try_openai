"""https://www.facebook.com/groups/stablediffusionthailand/permalink/1303061003642354/"""
import itertools
import math
import os.path
import random
import typing as typ

import diffusers
import torch
from controlnet_aux import HEDdetector
# !pip install transformers accelerate
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from read_lora import load_lora_weights_orig
from set_seed import seed_everything
from solve_77_limits import get_pipeline_embeds


def upscale() -> None:
    """Upscale the pictures."""
    pass


def controlnet_experiments() -> None:
    """Run controlNet."""
    models = [
        ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
        ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
        ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
        ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
        ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
        ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
        ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
        ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ]
    controlnet_names: typ.List[str] = [
        "control_v11p_sd15_normalbae",
        "control_v11p_sd15_canny",
        "control_v11p_sd15_mlsd",
        "control_v11p_sd15_scribble",
        "control_v11p_sd15_softedge",
        "control_v11p_sd15_seg",
        "control_v11p_sd15_lineart",
        "control_v11p_sd15s2_lineart_anime",
    ]
    schedulers: typ.List[typ.Tuple] = [
        ("LMSDiscreteScheduler", diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler),
        ("DDIMScheduler", diffusers.schedulers.scheduling_ddim.DDIMScheduler),
        (
            "DPMSolverMultistepScheduler",
            diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler),
        ("EulerDiscreteScheduler", diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler),
        ("PNDMScheduler", diffusers.schedulers.scheduling_pndm.PNDMScheduler),
        ("DDPMScheduler", diffusers.schedulers.scheduling_ddpm.DDPMScheduler),
        ("EulerAncestralDiscreteScheduler",
         diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler)
    ]
    loras = [
        (None, None),
        ("Night_scene_20230715120543", "../ai_files/loras/Night_scene_20230715120543.safetensors"),
        ("Oldsaigon-v1.0", "../ai_files/loras/Oldsaigon-v1.0.safetensors"),
        ("virginie_efira_v01", "../ai_files/loras/virginie_efira_v01.safetensors"),
    ]
    lora_multipliers = [1.0, 2.0]
    additional_prompts = [
        "a young mother is protecting her teenage daughter from scare tiger",
        "a handsome masculine topless terrorist with tatoo on arm is holding a sexy asian student hostage with knife",
    ]
    combined_list = list(itertools.product(
        models, controlnet_names, schedulers, loras, lora_multipliers, additional_prompts)
    )
    base_prompt: str = "a city picture of "
    negative_prompt: str = "(low quality, worst quality:1.4),"
    size_factor: float = 0.8
    seed: int = 8811

    random.shuffle(combined_list)
    for item in tqdm(combined_list, total=len(combined_list)):
        (model_name, model_path), controlnet_name, (scheduler_name, scheduler), \
            (lora_name, lora_path), lora_multiplier, add_prompt = item
        checkpoint: str = f"lllyasviel/{controlnet_name}"
        human_name: str = "boyzKhwan"
        out_dir: str = f"{human_name}/controlnet_experiments"

        init_image = load_image(f"{human_name}/sources/{human_name}.jpeg")
        width, height = init_image.size
        new_width, new_height = math.floor(width * size_factor / 8) * 8, math.floor(height * size_factor / 8) * 8

        init_image = init_image.resize((new_width, new_height))
        seed_everything(seed)
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(init_image, scribble=True)
        control_image.save(f"{human_name}/sources/control.png")
        controlnet = ControlNetModel.from_pretrained(checkpoint)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_path, controlnet=controlnet,
            requires_safety_checker=False,
            safety_checker=None
        )
        if lora_name != None:
            device: str = "cpu"
            print(f"{lora_name} : {device}")
            pipe = pipe.to(device)
            pipe = load_lora_weights_orig(pipe, lora_path, lora_multiplier, device, torch.float32)
        else:
            device: str = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"No lora : {device}")
            pipe = pipe.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)

        prompt: str = f"{base_prompt} {add_prompt}"
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)

        filename: str = f"{out_dir}/{model_name}_{controlnet_name}_{scheduler_name}_{lora_name}_{lora_multiplier}_{add_prompt[:20]}.png"
        print(f"{filename} is running")
        if not os.path.exists(filename):
            # generate image
            image = pipe(
                image=init_image,
                prompt_embeds=prompt_embeds,
                num_inference_steps=70,
                generator=generator,
                negative_prompt_embeds=negative_prompt_embeds,
            ).images[0].save(filename)
        else:
            print(f"{filename} is exists")


if __name__ == "__main__":
    controlnet_experiments()
