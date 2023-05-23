"""Text to image."""
import openai
import os

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

image_resp = openai.Image.create(prompt="two dogs playing chess, oil painting", n=4, size="512x512")
