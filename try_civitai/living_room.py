"""
Get safetensor from civitAI
https://civitai.com/models/43331/majicmix-realistic
"""
# Let's load the popular vermeer image
import typing as typ
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

#NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device.
# If you want this op to be added in priority during the prototype phase of this feature,
# please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix,
# you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
# WARNING: this will be slower than running natively on MPS.
# AppleInternal/Library/BuildRoots/2acced82-df86-11ed-9b95-428477786501/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:725:
# failed assertion `[MPSNDArray initWithDevice:descriptor:] Error: total bytes of NDArray > 2**32'
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
image = load_image(
    "./sources/laplace_filter_small.png"
)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

#❯ python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path controlNet/control_sd15_canny.pth --dump_path converted
control_model_name: str = "sd-controlnet-canny"
controlnet = ControlNetModel.from_pretrained(f"lllyasviel/{control_model_name}").to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    "sinkinai/majicMIX-realistic-v5",
    # "majicMIX-realistic-v5",
    safety_checker=None,
    controlnet=controlnet,
).to(device)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(12003)

# prompt: typ.List[str] = ["a tree", "red couch", "cozy", "new", "shinny"]
# negative_prompt: typ.List[str] = ["low quality", "bad", "old furniture", "dirty", "damage"]
# assert len(prompt) == len(negative_prompt)
prompt: str = "a tree, red couch, cozy, new, shinny"
negative_prompt: str = "low quality, bad, old furniture, dirty, damage"
for guidance_scale in range(0, 30):
    out_images = pipe(
        prompt, num_images_per_prompt=4,
        num_inference_steps=50, generator=generator, image=canny_image,
        guidance_scale=guidance_scale, negative_prompt=negative_prompt
    )
    for idx, image in enumerate(out_images.images):
        image.save(f"controlnet_images/{control_model_name}_{guidance_scale}_{idx}.png")
