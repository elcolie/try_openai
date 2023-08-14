"""Extract mp4 to frame by frame and then canny it and covert them by A.I."""
import math
import os.path

import cv2
import itertools
import os.path
import random

import diffusers
import torch
# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, StableDiffusionPipeline, AutoencoderKL, \
    StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything
from read_lora import load_lora_weights_orig
from solve_77_limits import get_pipeline_embeds
from PIL import Image
import numpy as np

human_name: str = "bood"
out_dir: str = f"{human_name}"
seed: int = 8811555
seed_everything(seed)
device: str = "cpu"
num_inference_steps: int = 150

def get_canny_image(filename: str) -> Image:
    """Return canny image."""
    image = load_image(filename)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def extract_raw_frames(mp4_file: str = "sources/bood.mp4") -> None:
    """Extract file to frame by frame."""
    vidcap = cv2.VideoCapture(mp4_file)
    success, image = vidcap.read()
    count = 0
    raw_frames_dir: str = "raw_frames"

    if not os.path.exists(raw_frames_dir):
        os.makedirs(raw_frames_dir)
    while success:
        frame_file: str = f"{raw_frames_dir}/frame{count}.jpg"
        if not os.path.exists(frame_file):
            cv2.imwrite(frame_file, image)  # save frame as JPEG file
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1
        else:
            print(f"{frame_file} exists")

def main() -> None:
    """Run main function."""

    base_prompt = "hires photo-realistic of concert band on the stage"
    additional_prompts = [
        "(masterpiece:1.2), best quality, PIXIV",
    ]
    negative_prompt: str = "(low quality, worst quality:1.4), bad hands, extra legs, drawing illustration, line art"
    strengths = [1, ]
    # guidance_scales = [round(0.1 * _, 3) for _ in range(70, 252, 2)]
    guidance_scales = [10, 20, 30, 40]
    # eta_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4, 6, 8, 10]
    eta_list = [1]
    image_path: str = "raw_frames/frame0.jpg"
    factor: float = 0.8
    init_image = get_canny_image(image_path)
    width, height = init_image.size
    new_width: int = math.floor(factor * width)
    new_height: int = math.floor(factor * height)
    init_image = init_image.resize((new_width, new_height))
    init_image.save("sources/bood_canny.png")

    models = [
        # ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
        # ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
        # ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
        # ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
        ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
        ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
        ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
        # ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
        # ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
    ]
    schedulers = [
        ("LMSDiscreteScheduler", diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler),
        ("DDIMScheduler", diffusers.schedulers.scheduling_ddim.DDIMScheduler),
        (
        "DPMSolverMultistepScheduler", diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler),
        ("UniPCMultistepScheduler", diffusers.UniPCMultistepScheduler),
        ("EulerDiscreteScheduler", diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler),
        ("PNDMScheduler", diffusers.schedulers.scheduling_pndm.PNDMScheduler),
        ("DDPMScheduler", diffusers.schedulers.scheduling_ddpm.DDPMScheduler),
        ("EulerAncestralDiscreteScheduler",
         diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler)
    ]
    loras = [
        (None, None),
        # ("virginie_efira_v01", "../ai_files/loras/virginie_efira_v01.safetensors"),
        # ("Night_scene_20230715120543", "../ai_files/loras/Night_scene_20230715120543.safetensors"),
        # ("Oldsaigon-v1.0", "../ai_files/loras/Oldsaigon-v1.0.safetensors"),
        # ("hayakawanagisa_lora-06", "../ai_files/lora/hayakawanagisa_lora-06.safetensors"),
        # ("nayeonlorashy", "../ai_files/lora/nayeonlorashy.safetensors"),
    ]
    lora_multipliers = [1]

    combined_list = list(itertools.product(
        models, schedulers, loras, lora_multipliers, strengths, guidance_scales, eta_list, additional_prompts)
    )

    # Shuffle the combined list
    random.shuffle(combined_list)

    for item in tqdm(combined_list, total=len(combined_list)):
        (model_name, model_dir), (scheduler_name, scheduler), (
        lora_name, lora_file), lora_multiplier, strength, guidance_scale, eta, add_prompt = item

        if lora_name != None:
            # TypeError: Trying to convert BFloat16 to the MPS backend, but it does not have support for that dtype.
            # device: str = "cpu"
            print(f"{lora_name} : {device}")
            controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny").to(device)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_dir, controlnet=controlnet,
                requires_safety_checker=False,
                safety_checker=None
            )
            pipe = pipe.to(device)
            pipe = load_lora_weights_orig(pipe, lora_file, lora_multiplier, device, torch.float32)
        else:
            # device: str = "mps" if torch.backends.mps.is_available() else "cpu"
            # device: str = "cpu"  # With this LoRA it can't run with mps. It raises RuntimeError: Invalid buffer size: 58.07 GB
            print(f"No lora : {device}")
            controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny").to(device)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_dir, controlnet=controlnet,
                num_inference_steps=num_inference_steps,
                requires_safety_checker=False,
                safety_checker=None
            )
            pipe = pipe.to(device)
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)

        generator = torch.Generator(device=device).manual_seed(seed)
        prompt: str = f"{base_prompt} {add_prompt}"
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filename: str = f"{out_dir}/{model_name}_{scheduler_name}_{lora_name}_{lora_multiplier}_{strength}_{guidance_scale}_{eta}_{base_prompt}_{add_prompt[:20]}.png".replace(
            " ", "_")
        print(f"{filename} is running")
        if not os.path.exists(filename):
            # generate image
            images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                          image=init_image, generator=generator, num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale).images
            images[0].save(filename)
        else:
            print(f"{filename} is exists")


if __name__ == "__main__":
    main()
