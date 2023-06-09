from itertools import combinations
import os
import typing as typ
from tqdm import tqdm
import torch
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

#NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device.
# If you want this op to be added in priority during the prototype phase of this feature,
# please comment on https://github.com/pytorch/pytorch/issues/77764.
# As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
# WARNING: this will be slower than running natively on MPS.

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

image = load_image(
    # "https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/input.png"
    # "c.jpeg",
    "control.png"
)

prompt = "bird, best quality, extremely detailed"
# prompt = "realistic woman"

processor_names = [
    "HEDdetector",
    "PidiNetDetector",
]
processor_checkpoints = [
    HEDdetector.from_pretrained('lllyasviel/Annotators'),
    PidiNetDetector.from_pretrained('lllyasviel/Annotators')
]
for processor_name, processor in zip(processor_names, processor_checkpoints):
    processor.netNetwork = processor.netNetwork.to(device)
    control_image = processor(image, safe=True)
    control_image.save("./images/control.png")

    base_checkpoints: typ.List[str] = [
        "lllyasviel/control_v11p_sd15_softedge",
        "lllyasviel/control_v11p_sd15_lineart",
        "lllyasviel/control_v11p_sd15s2_lineart_anime",
        "lllyasviel/control_v11p_sd15_canny",
        "lllyasviel/control_v11p_sd15_scribble",
        "lllyasviel/sd-controlnet-canny",
        "lllyasviel/sd-controlnet-depth",
        "lllyasviel/sd-controlnet-hed",
        "lllyasviel/sd-controlnet-normal",
        "lllyasviel/sd-controlnet-scribble",
    ]
    combinations_checkpoints = combinations(base_checkpoints, 2)
    for checkpoints in list(combinations_checkpoints):
        controlnet_list = [
            ControlNetModel.from_pretrained(checkpoint).to(device)
            for checkpoint in checkpoints
        ]
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet_list,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(181993)
        step: int = 30
        images = [control_image, control_image]
        image = pipe(prompt, images, num_inference_steps=step, generator=generator).images[0]

        image.save(f'images/image_out_{step}_{processor_name}_{"_AND_".join(checkpoints).replace("lllyasviel/", "")}.png')
