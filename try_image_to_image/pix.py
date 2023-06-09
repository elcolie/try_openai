"""Give order by sentence to let the A.I. do the editing."""
import torch
# https://github.com/timothybrooks/instruct-pix2pix
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

torch.manual_seed(44344)
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("mps" if torch.backends.mps.is_available() else "cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
url = "c.jpeg"
image = load_image(url)
prompt: str = "Transform to real human"
# image = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]
step: int = 30
scale: float = 1
image = pipe(prompt, image=image, num_inference_steps=step, image_guidance_scale=scale).images[0]
image.save(f"images/pix_{step}_{scale}.png")
