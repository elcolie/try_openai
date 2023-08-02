"""Try wood figure."""
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

image = load_image("sources/wood_figure.png")

image = openpose(image)
image.save("images/openpose_control_image.png")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose"
).to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    requires_safety_checker=False, safety_checker=None
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe("chef in the kitchen full body. face looking at the camera", image, num_inference_steps=20).images[0]

image.save('images/chef_pose_out.png')
