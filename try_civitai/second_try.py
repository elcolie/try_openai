"""
Get safetensor from civitAI
https://civitai.com/models/43331/majicmix-realistic
"""
from diffusers import StableDiffusionPipeline, StableDiffusionPix2PixZeroPipeline
import torch
torch.manual_seed(111)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

pipe = StableDiffusionPipeline.from_ckpt("majicmixRealistic_v5.safetensors", load_safety_checker=False).to(device)

prompt = "woman in bikini in the office"

negative_prompt: str = "bad hands, low quality"
for guidance_scale in [1, 2, 3, 4, 5]:
    for eta in [0.2, 0.4, 0.6, 0.8, 1.0]:
        result = pipe(
            prompt, num_inference_steps=30, num_images_per_prompt=8,
            guidance_scale=guidance_scale, negative_prompt=negative_prompt, eta=eta)
        for idx, image in enumerate(result.images):
            image.save(f"character_{guidance_scale}_{eta}_{idx}.png")
