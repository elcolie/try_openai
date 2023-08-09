"""https://huggingface.co/damo-vilab/text-to-video-ms-1.7b"""

import torch
import os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, TextToVideoSDPipeline
from diffusers.utils import export_to_video

from set_seed import seed_everything

seed: int = 888888
seed_everything(seed)
# RuntimeError: Conv3D is not supported on MPS
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
device: str = "cpu"

# TextToVideoSDPipeline
pipe = TextToVideoSDPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    # "../ai_directory/sd_xl_base_0.9", TypeError: argument of type 'NoneType' is not iterable
    # "../ai_directory/realisticVisionV50_v50VAE", RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 4, 16, 64, 64]
    # "runwayml/stable-diffusion-v1-5" RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 4, 16, 64, 64]
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
prompt = "a pretty girl in pink bikini walking on the beach. No watermark."
generator = torch.Generator(device=device).manual_seed(seed)
video_frames = pipe(
    prompt,
    num_inference_steps=500,
    num_frames=200,
    generator=generator
).frames
video_path = export_to_video(video_frames)
os.rename(video_path, "output.mp4")
