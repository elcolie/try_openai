"""
Try pure SDXL. https://civitai.com/models/122606
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/sdxl/dynavisionXLAllInOneStylized_beta0371Bakedvae.safetensors --from_safetensors --dump_path ai_directory/dynavisionXLAllInOneStylized_beta0371Bakedvae
"""
from diffusers import StableDiffusionXLPipeline
import torch
from set_seed import seed_everything

seed: int = 9999999
seed_everything(seed)

sdxl_models = [
    ("stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-xl-base-1.0"),
    ("dynavisionXLAllInOneStylized_beta0371Bakedvae", "../ai_directory/sdxl/dynavisionXLAllInOneStylized_beta0371Bakedvae")
]

device: str = "mps" if torch.backends.mps.is_available() else "cpu"
# device: str = "cpu"

prompt = "confused, looking around scared , pastel painting style,  a day in the life of a  citizen of Urumqi   ,   by  Carles Delclaux Is and Yaacov Agam beauty512"
negative_prompt = "No512 Neg512 neg_hands512, tiling poorly drawn out of frame stubby mutation mutated extra limbs extra legs extra arms disfigured deformed odd weird off putting out of frame bad anatomy double clones twins brothers same face repeated person long neck hat poorly drawn cropped text watermark signature logo split image copyright desaturated artifacts noise"

for model_name, model_path in sdxl_models:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        requires_safety_checker=False, safety_checker=None
    ).to(device)

    images = pipe(
        prompt=prompt, negative_prompt=negative_prompt
    ).images

    images[0].save(f"sdxl/{model_name}.png")
