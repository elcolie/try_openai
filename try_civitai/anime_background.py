"""
Experiment the real input background picture to anime.
https://civitai.com/models/35960/flat-2d-animerge
https://civitai.com/models/13910/thicker-lines-anime-style-lora-mix
https://civitai.com/models/87191/animechar-with-clothings-makai-tenshi-mix-or-sei-tenshi-djibril
"""
# Let's load the popular vermeer image
import os
import typing as typ
from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

#NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device.
# If you want this op to be added in priority during the prototype phase of this feature,
# please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix,
# you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
# WARNING: this will be slower than running natively on MPS.
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")
image = load_image(
    "./sources/46350.jpg"
)
output_dir: str = "anime_background"

image = np.array(image)
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(resized, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

control_nets: typ.List[str] = [
    "sd-controlnet-canny",
    "sd-controlnet-hed",
    "sd-controlnet-scribble",
]
for control_model_name in tqdm(control_nets):
    controlnet = ControlNetModel.from_pretrained(f"lllyasviel/{control_model_name}").to(device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "flat2DAnimerge",
        safety_checker=None,
        controlnet=controlnet,
        local_files_only=True,
        low_cpu_mem_usage=False
        # cache_dir="./flat2DAnimerge"
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    generator = torch.manual_seed(12003)

    prompt: str = "clean, two tables, 4 chairs, glass door, glass wall, dim light"
    negative_prompt: str = "low quality, dirty, damage, dark"
    num_images_per_prompt: int = 4
    for guidance_scale in range(0, 30, 2):
        # check existing file
        if not os.path.exists(f"{output_dir}/{control_model_name}_{guidance_scale}_{0}.png"):
            print("==============================")
            print(f"Running: {output_dir}/{control_model_name}_{guidance_scale}")
            out_images = pipe(
                prompt, num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=30, generator=generator, image=canny_image,
                guidance_scale=guidance_scale, negative_prompt=negative_prompt
            )
            for idx, image in enumerate(out_images.images):
                filename = f"{output_dir}/{control_model_name}_{guidance_scale}_{idx}.png"
                image.save(filename)
