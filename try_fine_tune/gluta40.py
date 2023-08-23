"""
gluta40 product.
- ผู้หญิง 35-40
- ถือสินค้าข้างซ้ายและรูปที่ถือข้างขวา หน้าตรง
- ผมดำ มัดผม และปล่อยผม
- พื้นหลังเรียบ
"""
import itertools
import os.path
import random
import string
import time
import typing as typ

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

import bb
from web_sdxl import seed_everything

seed_everything(42)


def generate_random_string(length):
    letters = string.ascii_letters
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(prompt.split(" "))
    count_negative_prompt = len(negative_prompt.split(" "))

    # create the tensor based on which prompt is longer
    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                          max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                       max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


model_path = "fine_tuned_models/sdxl-sarit"
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir: str = "gluta40"

age_prompts: typ.List[str] = [
    "young asian girl",
    "a photograph of a girl wearing a see-thru short roman style dress, beautiful asian mixed european woman face, beautiful eyes, black hair, looking down, hyper realistic and detailed, 16k",
]
hand_prompts: typ.List[str] = [
    "left hand holding a gluta40 jar one hand, right hand is behind her back",
    "right hand holding a gluta40 jar one hand, left hand is behind her back",
]
face_angle_prompts: typ.List[str] = [
    "straight face",
]
hair_prompts: typ.List[str] = [
    "black long tied hair",
    "black long hair",
]
background_prompts: typ.List[str] = [
    "no background, hold both hands, bad hands",
]
negative_prompt: str = "disfigured, disproportionate, bad anatomy, bad proportions, ugly, out of frame, mangled, asymmetric, cross-eyed, depressed, immature, stuffed animal, out of focus, high depth of field, cloned face, cloned head, age spot, skin blemishes, collapsed eyeshadow, asymmetric ears, imperfect eyes, unnatural, conjoined, missing limb, missing arm, missing leg, poorly drawn face, poorly drawn feet, poorly drawn hands, floating limb, disconnected limb, extra limb, malformed limbs, malformed hands, poorly rendered face, poor facial details, poorly rendered hands, double face, unbalanced body, unnatural body, lacking body, long body, cripple, cartoon, 3D, weird colors, unnatural skin tone, unnatural skin, stiff face, fused hand, skewed eyes, surreal, cropped head, group of people, too many fingers, bad hands, six fingers"
combined_list = list(itertools.product(age_prompts, hand_prompts, face_angle_prompts, hair_prompts, background_prompts))
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    age, hand, face_angle, hair, background = item
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prompt: str = f"{age[:30]}, {hand}, {face_angle}, {hair}, {background}"
    print(prompt)
    out_filename: str = f"{out_dir}/{prompt.replace(' ', '_')}"
    if not os.path.exists(f"{out_filename}_0.png"):
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, safety_checker=None,
                                                         requires_safety_checker=False)
        pipe.to(device)
        # prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)
        images = pipe(
            # prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            prompt=prompt, negative_prompt=negative_prompt,
            num_images_per_prompt=3, width=768,
            height=1024).images
        for idx, image in enumerate(images):
            image.save(f"{out_filename}_{idx}.png")
    else:
        print(f"{out_filename} exists")

bb.play_beep(440, 0.5)
time.sleep(1)
bb.play_beep(440, 0.5)
time.sleep(1)
bb.play_beep(440, 0.5)
