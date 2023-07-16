"""
https://www.facebook.com/groups/209358916852889/permalink/1046214066500699/
Further prompts
- see through
- strip bikini
- micro bikini
- nipple cover sexy
- sexy, white skin, straight face
- naked
Add multiple prompts
Result: Face is distorted
"""
import itertools
import math
import os
import random

# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from tqdm import tqdm

from set_seed import seed_everything, resize_for_condition_image

seed: int = 8888
seed_everything(seed)
out_dir: str = "sasithorn_bikini_beach"
negative_prompt: str = "nsfw, ng_deepnegative_v1_75t,badhandv4, (worst quality:2), (low quality:2), (normal quality:2), lowres,watermark, monochrome"
device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# init_image = init_image.resize((512, 512))
strengths = [1,]
guidance_scales = [round(0.2 * _, 3) for _ in range(36, 81, 1)]
# eta_list = [round(0.2 * _, 3) for _ in range(0, 11, 1)]
prompt = "best quality, masterpiece, (photorealistic:1.4), 1girl, light smile, a girl sitting on the beach bed. She crosses her legs and wearing 2 piece string bikini"
combined_list = list(itertools.product(strengths, guidance_scales))
random.shuffle(combined_list)
init_image = load_image("sources/sasithorn.jpeg")
width, height = init_image.size
size_factor: float = 0.6
new_width, new_height = math.floor(width * size_factor / 8) * 8, math.floor(height * size_factor / 8) * 8
# new_width, new_height = width, height
print(width, height)
print(new_width, new_height)
init_image = init_image.resize((new_width, new_height))

generator = torch.Generator(device=device).manual_seed(seed)
mask_image = load_image("sources/masked_sasithorn.png").resize((new_width, new_height))



def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = 1  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def main() -> None:
    """Run main function."""
    # strength = 1
    # guidance_scale = 7.5
    eta = 0
    for item in tqdm(combined_list):
        strength, guidance_scale = item
        add_prompt = prompt
        print(strength, guidance_scale, eta, add_prompt)
        filename: str = f"{out_dir}/{add_prompt}_{strength}_{guidance_scale}_{eta}_0.png"
        try:
            if not os.path.exists(filename):
                control_image = make_inpaint_condition(init_image, mask_image)

                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_inpaint",
                ).to(device)
                pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    # "runwayml/stable-diffusion-v1-5",  # Reference bad
                    "../ai_directory/majicmixRealistic_v6",  # Quite good
                    # "../ai_directory/MeinaV10",   # Anime
                    # "../ai_directory/perfectWorld_v4Baked",  # Ordinary
                    # "../ai_directory/chilloutmix_NiPrunedFp32Fix",  # It gives full bodysuit
                    controlnet=controlnet,
                    requires_safety_checker=False,
                    safety_checker=None
                ).to(device)
                pipe.requires_safety_checker = False
                pipe.safety_checker = None

                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

                # generate image
                result = pipe(
                    prompt=add_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=100,
                    generator=generator,
                    image=init_image,
                    mask_image=mask_image,
                    control_image=control_image,
                    num_images_per_prompt=2,
                    width=new_width,
                    height=new_height,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    # eta=eta,
                )
                for idx, image in enumerate(result.images):
                    filename: str = f"{out_dir}/{add_prompt}_{strength}_{guidance_scale}_{eta}_{idx}.png"
                    image.save(filename)
            else:
                print("File exists.")
        except Exception as err:
            print(err)
            print(f"{filename} is impossible")
            with open(f"{out_dir}/log.txt", "a") as file:
                file.write(str(err) + "\n")
                file.write(filename + "\n")
                file.write("====================================" + "\n")


if __name__ == "__main__":
    main()
