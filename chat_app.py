import os

import openai
from dotenv import load_dotenv

load_dotenv()


class ChatApp:
    """Chat instance."""
    def __init__(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.messages = [
            {"role": "system", "content": "You are a coding tutor bot to help user write and optimize python code."},
        ]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model=os.getenv("MODEL"),
            messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]
