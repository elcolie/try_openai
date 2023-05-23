import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history):
    user_message = history[-1][0]
    new_user_input_ids = tokenizer.encode(
        tokenizer.eos_token + user_message, return_tensors="pt"
    )

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor([]), new_user_input_ids], dim=-1)

    # generate a response
    response = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    ).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(response[0]).split("<|endoftext|>")
    response = [
        (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
    ]  # convert to tuples of list
    # Remove repeated pattern in the answer.
    no_duplicated_sentences = response[0][1].split(".")
    no_duplicated_list = list(dict.fromkeys(no_duplicated_sentences))
    sentence = ". ".join(no_duplicated_list)
    history[-1] = [user_message, sentence]
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
