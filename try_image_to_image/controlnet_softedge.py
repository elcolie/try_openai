import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import PidiNetDetector, HEDdetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

device = "cpu"  # PidiNetDetector is not support mps
checkpoint = "lllyasviel/control_v11p_sd15_softedge"

image = load_image(
    "https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/input.png"
)

prompt = "royal chamber with fancy bed"

# processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
control_image = processor(image, safe=True)
control_image.save("./images/control.png")

controlnet = ControlNetModel.from_pretrained(checkpoint).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

image.save('images/image_out.png')
