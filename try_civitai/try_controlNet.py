"""
Get safetensor from civitAI
https://civitai.com/models/43331/majicmix-realistic
"""
# Let's load the popular vermeer image
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
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")
image = load_image(
    # "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    "../try_image_to_image/c.jpeg"
)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    # "sinkinai/majicMIX-realistic-v5",
    "converted",
    safety_checker=None,
    controlnet=controlnet,
).to(device)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(1200)

prompt: str = "realistic woman, best quality"
negative_prompt: str = "low quality, bad hands"
num_images_per_prompt: int = 4
for guidance_scale in [
    # 0, 1, 1.5, 2, 2.5,
    3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
    out_images = pipe(
        prompt, num_inference_steps=20, generator=generator, image=canny_image,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale, negative_prompt=negative_prompt
    )
    for idx, image in enumerate(out_images.images):
        image.save(f"controlnet_images/{guidance_scale}_{idx}.png")
