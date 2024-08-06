import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fashion_mnist_model.h5')
    return model

model = load_model()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    img = ImageOps.grayscale(image)
    img = img.resize((28,28))
    img = np.array(img)
    img = img/255.0
    img = img.reshape(1,28,28,1)
    return img



st.title('Fashion MNIST Classifier')
st.write('Upload an image to do some classification')

uploaded_file = st.file_uploader('Choose an image....', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image= Image.open(uploaded_file)
    st.image(image,caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Image uploaded successfully!')

    if st.button('Run model'):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        st.write(f'Prediction: {predicted_class}')
        st.write(f'Confidence: {confidence:.2f}')