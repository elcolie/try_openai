import copy
import itertools
import random
import typing as typ

import torch
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.utils import load_image
from tqdm import tqdm

seed: int = 200
strengths: typ.List[float] = [0.4, 0.6, 0.8, 1.0]

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

source_image = load_image(
    "../try_text_to_image/bell/MeinaV10_EulerDiscreteScheduler_None_1.0_1_30_1_a white girl in pink bikini at the beach_(masterpiece:1.2), b.png")


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


_condition_image = resize_for_condition_image(source_image, 1024)


def controlnet_tile() -> None:
    """Controlnet approach."""
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)

    combined_list = list(itertools.product(models, strengths))
    random.shuffle(combined_list)

    for item in tqdm(combined_list, total=len(combined_list)):
        (model_name, model_path), strength = item
        controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile')
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline="stable_diffusion_controlnet_img2img",
            controlnet=controlnet,
            requires_safety_checker=False,
            safety_checker=None
        ).to(device)

        condition_image = copy.deepcopy(_condition_image)
        image = pipe(prompt="hires, a girl in pink bikini at the beach best quality",
                     negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                     image=condition_image,
                     controlnet_conditioning_image=condition_image,
                     width=condition_image.size[0],
                     height=condition_image.size[1],
                     strength=strength,
                     generator=torch.manual_seed(seed),
                     num_inference_steps=32,
                     ).images[0]

        image.save(f'controlnet_tile_upscale/{model_name}_{strength}.png')


if __name__ == "__main__":
    controlnet_tile()
