import streamlit as st
from PIL import Image
import numpy as np
from img_captioning_py_file import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#title
st.title("Image Captioning ")

#subtitle
st.markdown("## Automatic caption generator")

st.markdown("")

# image uploader
image = st.file_uploader(label="upload image here", type=['png','jpg','jpeg'])

if image is not None:
    input_image = Image.open(image)   #read image
    st.image(input_image) # display image

    with st.spinner(" AI is working ...."):
        img = np.array(input_image)
        img = np.resize(img, (299,299,3))
        img = img.reshape(1,299,299,3)
        img = preprocess_input(img)
        fe = extractor.predict(img, verbose=0)
        fe = fe.reshape(2048)
        result_1 = generate_desc(model=loaded_model, photo_fe=fe, inference=True)
        result_2 = beam_search_pred(model=loaded_model, pic_fe=fe, wordtoix=wordtoix, K_beams=3, log=False)
        result_3 = beam_search_pred(model=loaded_model, pic_fe=fe, wordtoix=wordtoix, K_beams=5, log=True)
        st.write(result_1)
        st.write(result_2)
        st.write(result_3)

    st.balloons()

else:
    st.write("Upload an image")
