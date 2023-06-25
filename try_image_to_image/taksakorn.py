"""Give order by sentence to let the A.I. do the editing."""
# https://github.com/timothybrooks/instruct-pix2pix
import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(44344)
model_id: str = "timbrooks/instruct-pix2pix"
url = "sources/taksakorn.jpeg"
num_images_per_prompt: int = 4
num_inference_steps: int = 50
prompt: str = "Transform to cartoon. Body shape is lean, bright and shinny background. High resolution"
negative_prompt: str = "Low resolution, black and white, dirty"
for image_guidance_scale in range(31):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
    pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image = load_image(url)
    if not os.path.exists(f"images/pix_{image_guidance_scale}_{0}.png"):
        print(f"Running images/pix_{image_guidance_scale}")
        results = pipe(prompt, negative_prompt=negative_prompt,
                       image=image, num_inference_steps=num_inference_steps,
                       image_guidance_scale=image_guidance_scale,
                       num_images_per_prompt=num_images_per_prompt)
        for idx, image in enumerate(results.images):
            image.save(f"images/pix_{image_guidance_scale}_{idx}.png")
