"""
https://www.facebook.com/photo/?fbid=230317606517173&set=gm.1042375796884526&idorvanity=209358916852889
Remove bikini.
"""
import itertools
import os
import random

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from tqdm import tqdm

from set_seed import seed_everything

seed: int = 8888
seed_everything(seed)
device: str = "cpu"
num_images_per_prompt: int = 20
num_inference_steps: int = 100
# image_guidance_scales = list(range(0, 30))
# guidance_scales = list(range(0, 30))
image_guidance_scales = [2]
guidance_scales = [8, 11]

combined_list = list(itertools.product(image_guidance_scales, guidance_scales))

# Shuffle the combined list
random.shuffle(combined_list)

model_id = "timbrooks/instruct-pix2pix"
prompt = "Replace her cloth with sexy bikini, shoulder bag, boob"
negative_prompt: str = "blur, bad quality, beach, bad shape, skinny"
source_image = load_image("sources/koch.jpeg")

for item in tqdm(combined_list):
    image_guidance_scale, guidance_scale = item
    if not os.path.exists(f'kotchakorn/output_{image_guidance_scale}_{guidance_scale}_{0}.png'):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        result = pipe(prompt, negative_prompt=negative_prompt,
                      image=source_image.resize((256, 256)),
                      num_inference_steps=num_inference_steps,
                      image_guidance_scale=image_guidance_scale,
                      guidance_scale=guidance_scale,
                      num_images_per_prompt=num_images_per_prompt)
        for idx, image in enumerate(result.images):
            image.save(f'kotchakorn/output_{image_guidance_scale}_{guidance_scale}_{idx}.png')
