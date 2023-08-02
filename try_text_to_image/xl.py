"""
https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9
>python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path sd_xl_base_0.9.safetensors --from_safetensors --dump_path sd_xl_base_0.9                                      ─╯
"""

import torch
from diffusers import DiffusionPipeline

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
pipe = DiffusionPipeline.from_pretrained("../ai_directory/sd_xl_base_0.9")  # StableDiffusionXLPipeline
pipe.to(device)
# RuntimeError: Python 3.11+ not yet supported for torch.compile
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt: str = "The beauty woman stand in the water, reflection of the lotus flowers in the water, RAW photo,(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, photo realistic"
negative_prompt: str = "bad shape, ugly, dirty"
num_inference_steps: int = 100
num_images_per_prompt: int = 4
result = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
              num_images_per_prompt=num_images_per_prompt)
for idx, image in enumerate(result.images):
    image.save(f"images/{idx}.png")
