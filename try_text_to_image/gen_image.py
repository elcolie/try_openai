from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("mps")

prompt = "rough collie half sit half stand on the chair"
image = pipe(prompt).images[0]

image.save("output.png")

