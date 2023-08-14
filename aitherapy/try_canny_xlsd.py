from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

from set_seed import seed_everything

seed: int = 9999999
seed_everything(seed)

num_inference_steps: int = 200
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
# device: str = "cpu"

prompt = "hires photo-realistic of concert band on the stage"
negative_prompt = "(low quality, worst quality:1.4), bad hands, extra legs, drawing illustration, line art"

image = load_image("../try_text_to_image/raw_frames/frame0.jpg")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
).to(device)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-xl-base-1.0",
    "../ai_directory/sdxl/dynavisionXLAllInOneStylized_beta0371Bakedvae",
    vae=vae,
    controlnet=controlnet,
    requires_safety_checker=False, safety_checker=None
).to(device)

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=num_inference_steps
    ).images

images[0].save(f"sdxl_images/canny_sdxl.png")
