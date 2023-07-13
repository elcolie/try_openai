"""
https://www.facebook.com/groups/209358916852889/permalink/1046931313095641/
Turn girl into anime.
Use this model.
https://civitai.com/models/23521/anime-pastel-dream
"""
import os.path
import random
import typing as typ
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything, resize_for_condition_image

seed: int = 8888
seed_everything(seed)
# RuntimeError: Invalid buffer size: 55.12 GB
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"
print(f"Device: {device}")
image = resize_for_condition_image(load_image("./sources/miw.jpeg"), 512)
out_dir: str = "anime_miw"

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble"
).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "../ai_directory/animePastelDream",
    safety_checker=None,
    controlnet=controlnet,
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(seed)

prompt: str = "masterpiece, best quality, ultra-detailed, illustration, (1girl), small scorpion tatoo on breast"
negative_prompt: str = "low quality, bad hands"
num_images_per_prompt: int = 1
guidances: typ.List[float] = [
    0, 1, 1.5, 2, 2.5, 3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7,
    8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5
]
random.shuffle(guidances)
for guidance_scale in tqdm(guidances):
    filename: str = f"{out_dir}/{guidance_scale}_0.png"
    if not os.path.exists(filename):
        out_images = pipe(
            prompt, num_inference_steps=100, generator=generator, image=canny_image,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale, negative_prompt=negative_prompt
        )
        for idx, image in enumerate(out_images.images):
            image.save(f"{out_dir}/{guidance_scale}_{idx}.png")
