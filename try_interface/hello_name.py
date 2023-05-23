import gradio as gr
import numpy as np
import time

def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


def chatbot_vs_human(name: str):
    return "Hello " + name + "!"

def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(word, desc="Reversing"):
        time.sleep(0.25)
        new_string = letter + new_string
    return new_string


def echo(name, request: gr.Request):
    if request:
        print("Request headers dictionary:", request.headers)
        print("IP address:", request.client.host)
    return name

def video_display(input_img):
    return input_img


def main() -> None:
    """Run main function."""
    io = gr.Interface(video_display, gr.Video(), "video").launch()

    # demo = gr.Interface(slowly_reverse, gr.Text(), gr.Text())
    # demo.queue(concurrency_count=10).launch(
    #     # share=True
    # )


if __name__ == "__main__":
    main()
