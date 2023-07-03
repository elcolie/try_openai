"""
Overcome the limit by using chunk.
https://github.com/huggingface/diffusers/issues/2136
"""
from diffusers import StableDiffusionPipeline
import torch
import random

# 1. load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


pipe.enable_sequential_cpu_offload() # my graphics card VRAM is very low


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


prompt = (22 + random.randint(1, 10)) * "a photo of an astronaut riding a horse on mars"
negative_prompt = (22 + random.randint(1, 10)) * "some negative texts"

print("Our inputs ", prompt, negative_prompt, len(prompt.split(" ")), len(negative_prompt.split(" ")))

prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, "cuda")

image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]

image.save("done.png")
