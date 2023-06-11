"""
Get safetensor from civitAI
https://civitai.com/models/43331/majicmix-realistic
"""
from diffusers import StableDiffusionPipeline, StableDiffusionPix2PixZeroPipeline
import torch
torch.manual_seed(111)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

pipe = StableDiffusionPipeline.from_ckpt("majicmixRealistic_v5.safetensors", safety_checker=None).to(device)

prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

result = pipe(prompt, num_inference_steps=900, num_images_per_prompt=4)
for idx, image in enumerate(result.images):
    image.save(f"character_{idx}.png")
