"""
https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9
>python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path sd_xl_base_0.9.safetensors --from_safetensors --dump_path sd_xl_base_0.9                                      ─╯
"""

from diffusers import DiffusionPipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
pipe = DiffusionPipeline.from_pretrained("sd_xl_base_0.9")
pipe.to(device)

# RuntimeError: Python 3.11+ not yet supported for torch.compile
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save(f"images/astronaut.png")
