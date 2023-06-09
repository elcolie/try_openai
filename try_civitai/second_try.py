"""
Get safetensor from civitAI
"""
import os
import typing as typ
import itertools
import random
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm

from read_lora import load_lora_weights

torch.manual_seed(111)
device = "mps" if torch.backends.mps.is_available() else "cpu"
multipliers: typ.List[float] = list(range(1, 11))
guidances: typ.List[float] = list(range(1, 11))
etas: typ.List[float] = list(range(1, 11))
output_dir: str = "zzz"

# Combine the lists into a single list of tuples
combined_list = list(itertools.product(multipliers, guidances, etas))


# Shuffle the combined list
random.shuffle(combined_list)

for item in tqdm(combined_list, desc="Overall"):
    _multiplier, _guidance, _eta = item
    multiplier: float = 0.2 * _multiplier
    guidance_scale: float = 0.2 * _guidance
    eta: float = 0.2 * _eta
    # Check existing file
    if not os.path.isfile(f"{output_dir}/character_{multiplier}_{guidance_scale}_{eta}_0.png"):
        print(f"Start working on character_{multiplier}_{guidance_scale}_{eta}")
        # https://civitai.com/models/43331/majicmix-realistic
        model_base = "majicmixRealistic_v6.safetensors"
        pipe = StableDiffusionPipeline.from_ckpt(
            model_base,
            load_safety_checker=False)

        # https://civitai.com/models/82098/add-more-details-detail-enhancer-tweaker-lora
        pipe = load_lora_weights(pipe, "../ai_files/more_details_lora.safetensors", multiplier, device)
        pipe.to(device)
        prompt = """
        best quality, 8K, HDR, highres, blurry background, bokeh:1.3, Photography, (RAW photo:1.2, photorealistic:1.4, masterpiece:1.3, best quality:1.3, ultra highres:1.2), (((pin light, backlighting))), (depth of field), sharp focus:1.4, (camera from distance), (super_detail, hyper_detail, finely_detailed) ADDBASE blue sky ADDROW blue see ADDROW sand beach BREAK 1woman standing on sand beach, front view person. full body, ((((mature woman, solo, glamorous)))), (((gigantic breasts:0.7, narrow waist, oiledwet, large ass, thin thighs))) BREAK high detailed skin, delicate, beautiful skin, solid circle eyes, detailed beautiful face, beautiful detailed, (detailed eyes), (detailed facial features), beautiful and clear eyes, detail eye pupil, lipgloss, blush, cheek, long hair, black hair, (((expression_face, hair over one eye, beautiful fingers, beautiful hands))) BREAK bikini, earrings, necklace, sandals
        """
        negative_prompt: str = "Negative prompt: ((((BadNegAnatomyV1-neg, badhandv4)))), EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), ((((skin spots, acnes, skin blemishes, age spot)))), (((water mark, username, signature, text))), (((extra fingers, extra legs, extra foot, missing body, missing face, missing arms, missing hands, missing fingers, missing legs, missing feet, missing toe, strange fingers, bad hands, fewer fingers, extra digit, fewer digits)))"
        result = pipe(
            prompt, num_inference_steps=50, num_images_per_prompt=10, eta=eta,
            guidance_scale=guidance_scale, negative_prompt=negative_prompt)
        for idx, image in enumerate(result.images):
            output_filename: str = f"{output_dir}/character_{multiplier}_{guidance_scale}_{eta}_{idx}.png"
            image.save(output_filename)
    else:
        pass
