"""Give order by sentence to let the A.I. do the editing."""
# https://github.com/timothybrooks/instruct-pix2pix
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("mps")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
url = "c.jpeg"
image = load_image(url)
prompt: str = "Turn drawing into realistic colored human."
image = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1).images[0]
image.save(f"images/pix.png")
