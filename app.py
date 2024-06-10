import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_pipeline():
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )
        pipe.to("cuda")
        return pipe
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

st.title("Text to Image Generator")

# Load the model
pipe = load_pipeline()

# User input
prompt = st.text_input("Enter your prompt:", "Pixar style little boy holding ice cream cone smiling super detailed unreal engine 5 realistic intricate elegant highly detailed digital painting artstation concept art by Mark Brooks and Brad Kunkle detailed")

if st.button("Generate Image"):
    if pipe is None:
        st.error("Model failed to load. Please try again later.")
    else:
        with st.spinner('Generating...'):
            try:
                images = pipe(prompt=prompt).images[0]
                images.save("image.png")
                st.image(images, caption='Generated Image')
            except Exception as e:
                st.error(f"Error generating image: {e}")
