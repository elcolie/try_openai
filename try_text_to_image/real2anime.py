"""
Convert real picture to anime style.
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/abased_v10.safetensors --from_safetensors --dump_path ai_directory/abased_v10
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/exquisiteDetails_v10.safetensors --from_safetensors --dump_path ai_directory/exquisiteDetails_v10
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/kawaiiRealisticAnime_a03.safetensors --from_safetensors --dump_path ai_directory/kawaiiRealisticAnime_a03
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/lyricalVivian_v10.safetensors --from_safetensors --dump_path ai_directory/lyricalVivian_v10
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ai_files/animeArtDiffusionXL_alpha3.safetensors --from_safetensors --dump_path ai_directory/animeArtDiffusionXL_alpha3
"""
import itertools
import typing as typ
import math
import os.path
import random
import cv2
import diffusers
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from tqdm import tqdm
from PIL import Image
import numpy as np

from read_lora import load_lora_weights_orig
from set_seed import seed_everything

seed: int = 111
seed_everything(seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"
moaw_dir: str = "/Users/sarit/study/try_openai/try_civitai/sources/moaw"
out_dir: str = "real2anime"

def get_canny_image(filename: str) -> Image:
    """Return canny image."""
    image = load_image(filename)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

base_prompts: typ.List[str] = [
    "painting",
    "drawing",
    "sketching",
    "oil paint",
    "water colour picture",
    "anime",
]

pictures = [
    ("the tables and chairs in the cafe. the wall has fireplace over fireplace is tv", f"{moaw_dir}/9.jpg"),
    # ("inside cafe. through glass door and glass window", f"{moaw_dir}/46350.jpg"),
    # ("cafe with many tables and chairs. Wall is decorated with pictures", f"{moaw_dir}/46351.jpg"),
    # ("a sunset at the sea", f"{moaw_dir}/46813.jpg"),
]
models = [
    ("abased_v10", "../ai_directory/abased_v10"),
    # ("animeArtDiffusionXL_alpha3", "../ai_directory/animeArtDiffusionXL_alpha3"), Can't user contorlnet
    ("animePastelDream", "../ai_directory/animePastelDream"),
    ("anythingV3_fp16", "../ai_directory/anythingV3_fp16"),
    ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
    ("exquisiteDetails_v10", "../ai_directory/exquisiteDetails_v10"),
    ("flat2DAnimerge", "../ai_directory/flat2DAnimerge"),
    ("henmixrealV10_henmixrealV10", "../ai_directory/henmixrealV10_henmixrealV10"),
    ("kawaiiRealisticAnime_a03", "../ai_directory/kawaiiRealisticAnime_a03"),
    ("lyricalVivian_v10", "../ai_directory/lyricalVivian_v10"),
    ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),
    ("MeinaV10", "../ai_directory/MeinaV10"),
    ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),
    ("realdosmix", "../ai_directory/realdosmix"),
    ("realisticVisionV40_v40VAE", "../ai_directory/realisticVisionV40_v40VAE"),
    ("realisticVisionV50_v50VAE", "../ai_directory/realisticVisionV50_v50VAE"),
    # ("sd_xl_base_0.9", "../ai_directory/sd_xl_base_0.9"), # Can't use controlnet
]
# schedulers = [
#     ("LMSDiscreteScheduler", diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler),
#     ("DDIMScheduler", diffusers.schedulers.scheduling_ddim.DDIMScheduler),
#     ("DPMSolverMultistepScheduler", diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler),
#     ("EulerDiscreteScheduler", diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler),
#     ("PNDMScheduler", diffusers.schedulers.scheduling_pndm.PNDMScheduler),
#     ("DDPMScheduler", diffusers.schedulers.scheduling_ddpm.DDPMScheduler),
#     ("EulerAncestralDiscreteScheduler",
#      diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler)
# ]
# loras = [
#     (None, None),
#     ("Night_scene_20230715120543", "../ai_files/loras/Night_scene_20230715120543.safetensors"),
#     ("Oldsaigon-v1.0", "../ai_files/loras/Oldsaigon-v1.0.safetensors"),
#     ("virginie_efira_v01", "../ai_files/loras/virginie_efira_v01.safetensors"),
# ]
guess_modes = [
    # True, NotImplementedError: The operator 'aten::logspace.out' is not currently implemented for the MPS device
    False
]
# lora_multipliers = [1.0, 2.0, 3, 0]
combined_list = list(itertools.product(
    base_prompts,
    pictures,
    models,
    # schedulers,
    guess_modes,
    # loras,
    # lora_multipliers
    )
)
random.shuffle(combined_list)

for item in tqdm(combined_list, total=len(combined_list)):
    base, (desc, image_path), (model_name, model_path), guess_mode = item
    prompt: str = f"{base}, {desc}"
    filename: str = f"{out_dir}/{base}_{desc}_{model_name}_{guess_mode}.png"
    if not os.path.exists(filename):
        print(filename)
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny").to(device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_path, controlnet=controlnet,
            requires_safety_checker=False,
            safety_checker=None
        )

        # pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
        # if lora_name != None:
        #     device: str = "cpu"
        #     print(f"{lora_name} : {device}")
        #     pipe = pipe.to(device)
        #     pipe = load_lora_weights_orig(pipe, lora_file, lora_multiplier, device, torch.float32)
        # else:
        #     device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        #     print(f"No lora : {device}")
        #     pipe = pipe.to(device)
        pipe = pipe.to(device)
        factor: float = 0.5
        init_image = get_canny_image(image_path)
        width, height = init_image.size
        new_width: int = math.floor(factor * width)
        new_height: int = math.floor(factor * height)
        init_image = init_image.resize((new_width, new_height))
        images = pipe(prompt=prompt,
                      negative_prompt="realistic",
                      image=init_image,
                      guidance_scale=7.5,
                      guess_mode=guess_mode).images
        images[0].save(filename)
    else:
        print(f"{filename} exist")
