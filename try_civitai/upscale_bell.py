"""Bell."""
import itertools
import os.path
import random
import typing as typ

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from tqdm import tqdm
from set_seed import seed_everything

seed: int = 200
seed_everything(seed)
strengths: typ.List[float] = [1.0]

models = [
    ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
    ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
    # ("MeinaV10", "../ai_directory/MeinaV10"),  # Anime
    ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
    ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
    ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
]

source_images = [
    ("b", "/Users/sarit/study/GFPGAN/results/restored_imgs/output.png")
]


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


def controlnet_tile() -> None:
    """Controlnet approach."""
    # device: str = "mps" if torch.backends.mps.is_available() else "cpu"  # Error: total bytes of NDArray > 2**32
    device: str = "cpu"
    print(device)

    combined_list = list(itertools.product(models, strengths, source_images))
    random.shuffle(combined_list)

    for item in tqdm(combined_list, total=len(combined_list)):
        (model_name, model_path), strength, (image_name, source_image) = item
        filename: str = f'bell_inpaint/controlnet_tile_upscale/{model_name}_{strength}_{image_name}.png'
        print(filename)
        if not os.path.exists(filename):
            controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile')
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_path,
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=controlnet,
                requires_safety_checker=False,
                safety_checker=None
            ).to(device)
            condition_image = resize_for_condition_image(load_image(source_image), 1024)
            image = pipe(prompt="The beauty woman stand in the water, reflection of the lotus flowers in the water, RAW photo,(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, photo realistic",
                         negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                         image=condition_image,
                         controlnet_conditioning_image=condition_image,
                         width=condition_image.size[0],
                         height=condition_image.size[1],
                         strength=strength,
                         generator=torch.manual_seed(seed),
                         ).images[0]
            image.save(filename)
        else:
            print("exists")


if __name__ == "__main__":
    controlnet_tile()
