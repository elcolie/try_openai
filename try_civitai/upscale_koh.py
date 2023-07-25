import os

import torch
from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils import load_image
from tqdm import tqdm

# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"
print(device)
seed: int = 200


def x4() -> None:
    """x4 upscaler."""
    # load model and scheduler
    path_dir: str = "koh/next"
    files = os.listdir(path_dir)
    for file in tqdm(files):
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, )
        pipeline = pipeline.to(device)

        prompt = "A masculine topless guy holding a bag"

        low_res_img = load_image(f"{path_dir}/{file}")
        upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
        upscaled_image.save(f"{path_dir}/{file}_x4.png")


if __name__ == "__main__":
    x4()
