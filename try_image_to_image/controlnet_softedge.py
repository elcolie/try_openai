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
device = "cpu"  # PidiNetDetector is not support mps

image = load_image(
    # "https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/input.png"
    # "c.jpeg",
    "control.png"
)

prompt = "bird"
# prompt = "A realistic asian woman with cloth"

# processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
processor.netNetwork = processor.netNetwork.to(device)
control_image = processor(image, safe=True)
control_image.save("./images/control.png")
# control_image = image


# checkpoint: str = "lllyasviel/control_v11p_sd15_softedge"
checkpoint: str = "lllyasviel/control_v11p_sd15_lineart"
# checkpoint: str = "lllyasviel/control_v11p_sd15s2_lineart_anime"
controlnet = ControlNetModel.from_pretrained(checkpoint).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

generator = torch.manual_seed(181993)
step: int = 30
image = pipe(prompt, num_inference_steps=step, generator=generator, image=control_image).images[0]

image.save(f'images/image_out_{step}_2.png')
