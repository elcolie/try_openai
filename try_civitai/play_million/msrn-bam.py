import torch
from diffusers.utils import load_image
from super_image import MsrnModel, ImageLoader

device: str = "mps" if torch.backends.mps.is_available() else "cpu"
# from PIL import Image
# import requests
# url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = load_image("/Users/sarit/million/IMG_5696.JPG")

model = MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=2).to(device)  # scale 2, 3 and 4 models available
inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(preds, './output/scaled_2x.png')  # save the output 2x scaled image to `./scaled_2x.png`
ImageLoader.save_compare(inputs, preds,
                         './output/scaled_2x_compare.png')  # save an output comparing the super-image with a bicubic scaling
