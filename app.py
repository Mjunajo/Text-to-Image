import gradio as gr
from diffusers import DiffusionPipeline
import torch

# Load the model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    images = pipe(prompt=prompt).images[0]
    return images

iface = gr.Interface(fn=generate_image, inputs="text", outputs="image", title="Text to Image Generator", description="Enter a prompt to generate an image")
iface.launch()
