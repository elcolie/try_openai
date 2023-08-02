"""Try diffusion letter."""
import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from controlnet_aux import LineartDetector
from set_seed import seed_everything

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
seed: int = 888
seed_everything(seed)
# NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device.
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"
checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
models = [
    ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),
    ("abased_v10", "../ai_directory/abased_v10"),
    # ("animeArtDiffusionXL_alpha3", "../ai_directory/animeArtDiffusionXL_alpha3"),  # Can't use with controlnet
    ("animePastelDream", "../ai_directory/animePastelDream"),
    ("anythingV3_fp16", "../ai_directory/anythingV3_fp16"),
    ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    ("exquisiteDetails_v10", "../ai_directory/exquisiteDetails_v10"),
    ("flat2DAnimerge", "../ai_directory/flat2DAnimerge"),
    ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
    ("kawaiiRealisticAnime_a03", "../ai_directory/kawaiiRealisticAnime_a03"),
    ("lyricalVivian_v10", "../ai_directory/lyricalVivian_v10"),
    ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),
    ("MeinaV10", "../ai_directory/MeinaV10"),
    ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),
    ("realdosmix", "../ai_directory/realdosmix"),
    ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
    # ("sd_xl_base_0.9", "../ai_directory/sd_xl_base_0.9"), # Can't use with controlnet
]
image = load_image("lineart/str_pop.png")
# image = image.resize((512, 512))

prompt = "electronic circuit and neural network"
processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

control_image = processor(image)
control_image.save("./lineart/control.png")

for item in tqdm(models, total=len(models)):
    (model_name, model_path) = item
    filename: str = f"lineart/outputs/{model_name}.png"
    if not os.path.exists(filename):
        print(f"{filename} is running")
        try:
            controlnet = ControlNetModel.from_pretrained(checkpoint).to(device)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_path, controlnet=controlnet,
                requires_safety_checker=False,
                safety_checker=None
            ).to(device)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            generator = torch.manual_seed(seed)
            image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
            image.save(filename)
            print("saved")
        except Exception as err:
            print(err)
    else:
        print(f"{filename} exists")
