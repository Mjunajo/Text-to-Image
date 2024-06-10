import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Load the model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

st.title("Text to Image Generator")

# User input
prompt = st.text_input("Enter your prompt:", "Handsome photo of Babar Azam, lifelike, super highly detailed, professional digital painting, artstation, concept art, smooth, sharp focus, extreme illustration, Unreal Engine 5, Photorealism, HD quality, 8k")

if st.button("Generate Image"):
    with st.spinner('Generating...'):
        images = pipe(prompt=prompt).images[0]
        images.save("image.png")
        st.image(images, caption='Generated Image')
