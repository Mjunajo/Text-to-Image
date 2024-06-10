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
prompt = st.text_input("Enter your prompt:", "Handsome photo of Babar Azam, lifelike, super highly detailed, professional digital painting, artstation, concept art, smooth, sharp focus, extreme illustration, Unreal Engine 5, Photorealism, HD quality, 8k")

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
