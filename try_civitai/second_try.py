"""
Get safetensor from civitAI
https://civitai.com/models/43331/majicmix-realistic
"""
from diffusers import StableDiffusionPipeline
import torch
torch.manual_seed(111)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

pipe = StableDiffusionPipeline.from_ckpt("majicmixRealistic_v5.safetensors", load_safety_checker=False).to(device)

prompt = "A photo of rough collie, best quality"

negative_prompt: str = "low quality"
guidance_scale = 1
eta = 0.0
result = pipe(
    prompt, num_inference_steps=30, num_images_per_prompt=8,
    guidance_scale=1, negative_prompt=negative_prompt)
for idx, image in enumerate(result.images):
    image.save(f"character_{guidance_scale}_{eta}_{idx}.png")
