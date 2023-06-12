"""
https://huggingface.co/docs/diffusers/using-diffusers/img2img
"""
import typing as typ
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

device = "mps" if torch.backends.mps.is_available() else "cpu"
generator = torch.Generator(device=device).manual_seed(1024)
prompt: typ.List[str] = ["A woman", "realistic color", "photo", "high quality"]
strengths = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
guidance_scales = [_ * 0.2 for _ in range(1, 11)]
negative_prompt: typ.List[str] = ["bad hands", "blur", "grey", "black and white"]
num_inference_steps = 500
print(f"Total run: {len(strengths) * len(guidance_scales)}")

for strength in strengths:
    for guidance_scale in guidance_scales:
        pipe = StableDiffusionImg2ImgPipeline.from_ckpt("majicmixRealistic_v5.safetensors",
                                                        load_safety_checker=False).to(device)
        url = "../try_image_to_image/c.jpeg"
        init_image = load_image(url)

        # image= pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
        image = pipe(
            prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale,
            generator=generator, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
        image.save(f"images/3rd_{strength}_{guidance_scale}.png")
