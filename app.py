import numpy as np
import streamlit as st
import PIL
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""## Loading Trained Model"""
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('colormodel_trained_90.h5') 

"""## Initializing Color Classes for Prediction"""

# Mapping the Color Index with the respective 11 Classes (More Explained in RGB Color Classifier ML Model jupyter notebook)
color_dict={
    0 : 'Red',
    1 : 'Green',
    2 : 'Blue',
    3 : 'Yellow',
    4 : 'Orange',
    5 : 'Pink',
    6 : 'Purple',
    7 : 'Brown',
    8 : 'Grey',
    9 : 'Black',
    10 : 'White'
}

#predicting from loaded trained_model
def predict_color(Red, Green, Blue):
    rgb = np.asarray((Red, Green, Blue)) #rgb tuple to numpy array
    input_rgb = np.reshape(rgb, (-1,3)) #reshaping as per input to ANN model
    color_class_confidence = model.predict(input_rgb) # Output of layer is in terms of Confidence of the 11 classes
    color_index = np.argmax(color_class_confidence, axis=1) #finding the color_class index from confidence
    color = color_dict[int(color_index)]
    return color


# display image with the size and rgb color
def display_image():
    img = Image.new("RGB", (200, 200), color=(Red,Green,Blue))
    img = ImageOps.expand(img, border=1, fill='black')  # border to the img
    st.image(img, caption='RGB Color')

import streamlit.components.v1 as components
def github_gist(gist_creator, gist_id, height=600, scrolling=True):
    components.html(
        f"""
    <script src="https://gist.github.com/{gist_creator}/{gist_id}.js">
    </script>
    """,
        height=height,
        scrolling=scrolling,
    )

st.header("Select RGB values")

Red = st.slider( label="RED value: ", min_value=0, max_value=255, value=0, key="red")
Green = st.slider(label="GREEN value: ", min_value=0, max_value=255, value=0, key="green")
Blue = st.slider(label="BLUE value: ", min_value=0, max_value=255, value=0, key="blue")

st.write('Red: {}, Green: {}, Blue: {}'.format(Red, Green, Blue))
display_image()
result = ""
if st.button("Predict"):
    result = predict_color(Red, Green, Blue)
    st.success('The Color is {}!'.format(result))

st.markdown(hide_st_style, unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.info(
        "**RGB Color Classifier** can Predict upto 11 Distinct Color Classes based on the RGB input by User from the sliders\n\n"
        "The 11 Classes are *Red, Green, Blue, Yellow, Orange, Pink, Purple, Brown, Grey, Black and White*\n\n"
        "This app is created and maintained by "
    )
st.sidebar.title("Contribute")
st.sidebar.info(
    "You are very **Welcome** to contribute your awesome comments, questions or suggestions ")
#st.title("RGB Color Classifier")

# Render a gist
github_gist('tc87', '9382eafdb6eebde0bca0c33080d54b58')

#st.markdown(title_html, unsafe_allow_html=True) #Title rendering



