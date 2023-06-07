"""Run on CPU M2 not support MPS. Give image and prompt describe the output."""
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image

image = load_image("c.jpeg")
image = np.array(image)
control_image = Image.fromarray(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    # "lllyasviel/control_v11p_sd15_openpose",
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(31)
prompt: str = "A asian woman wearing tank top."
image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]

image.save('images/out.png')
