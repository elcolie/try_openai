from diffusers import StableDiffusionPipeline

model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, requires_safety_checker=False)
pipe.to("mps")

prompt: str = "orange astronaut riding a horse on mars, high resolution, high definition"
image = pipe(prompt=prompt).images[0]
image.save(f"{prompt.replace(' ', '_')}.png")
