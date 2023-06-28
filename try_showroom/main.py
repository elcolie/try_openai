import gradio as gr
import random

# Define a function to generate random images
def generate_images(num_images):
    images = []
    for _ in range(num_images):
        # Generate random image data (replace this with your own image generation logic)
        image_data = [random.randint(0, 255) for _ in range(784)]  # 28x28 image
        images.append(image_data)
    return images

# Create a callback function to generate new images on button click
def generate_images_callback(num_images):
    return generate_images(num_images)

# Create a Gradio interface
def create_gradio_interface():
    # Generate initial set of random images
    images = generate_images(12)

    # Define the layout of the gallery
    gallery = gr.inputs.ImageGroup(images=images, label=None, rows=3, cols=4, display_fn=None)

    # Create a button to generate new images
    generate_button = gr.inputs.Button(text="Generate", label=None, type="primary")

    # Combine the gallery and button in a Gradio interface
    interface = gr.Interface([gallery, generate_button], generate_images_callback, title="Image Gallery", theme="compact")

    return interface

# Run the Gradio interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
