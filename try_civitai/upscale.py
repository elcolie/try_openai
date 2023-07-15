import torch
from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils import load_image

device: str = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
seed: int = 200


def x4() -> None:
    """x4 upscaler."""
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, )
    pipeline = pipeline.to(device)

    # let's download an  image
    # url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    # response = requests.get(url)
    # low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    # low_res_img = low_res_img.resize((128, 128))
    low_res_img = load_image("sources/sasithorn.jpeg")

    prompt = "a girl sitting in the park"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    upscaled_image.save("upsampled_sasithorn.png")


if __name__ == "__main__":
    x4()
