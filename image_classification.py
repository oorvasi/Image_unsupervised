import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


st.title('Image Classification')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

image = Image.open(img_file_buffer)
img_array = np.array(image)

if image is not None:
    st.image(
        image,
        caption=f"You amazing image has shape {img_array.shape[0:2]}",
        use_column_width=True,
    )