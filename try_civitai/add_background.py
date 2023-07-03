"""Add background experiment."""
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from set_seed import seed_everything

seed_everything(200)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running: {device}")
file: str = "./sources/eunho.png"
masked_image: str = "./sources/eunho_.png"
image = load_image(file)
mask_image = load_image(masked_image)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    safety_checker=None
)
prompt = "computer"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image, num_inference_steps=30).images[0]
image.save("./nat/eunho_with_background.png")
