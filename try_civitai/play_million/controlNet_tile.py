import torch
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.utils import load_image

from set_seed import seed_everything

seed: int = 888
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"

seed_everything(seed)


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile')
pipe = DiffusionPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    "../flat2DAnimerge",
    custom_pipeline="stable_diffusion_controlnet_img2img",
    controlnet=controlnet,
    safety_checker=None).to(device)

source_image = load_image("/Users/sarit/million/IMG_5696.JPG")
condition_image = resize_for_condition_image(source_image, 1024)
image = pipe(prompt="best quality, high resolution, clean, medium light",
             negative_prompt="blur, low resolution, bad anatomy, bad hands, cropped, worst quality, sweat",
             image=condition_image,
             controlnet_conditioning_image=condition_image,
             width=condition_image.size[0],
             height=condition_image.size[1],
             strength=1.0,
             generator=torch.manual_seed(seed),
             num_inference_steps=32,
             ).images[0]

image.save('output/output.png')
