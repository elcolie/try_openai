from diffusers import StableDiffusionPipeline
import torch
torch.manual_seed(111)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")

prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

image = pipe(prompt, num_inference_steps=100).images[0]
image.save("character.png")
