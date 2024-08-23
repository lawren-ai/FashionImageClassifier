import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np




model_path = "C:/Users/HP/Desktop/ML projects/Fashion_Mnist_model/trained_fashion_mnist_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

#Define class labels for Fashion MNIST dataset
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


#Function to preproceww the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert("L")
    img.array = np.array(img) / 255
    img_array = img.array.reshape((1, 28, 28, 1))
    return img_array

#Building the streamlit app
st.title("Fashion Images Classifier")

uploaded_image = st.file_uploader("Upload an Image: ", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100,100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            #preprocess image
            img_array = preprocess_image(uploaded_image)
            #make a prediction using the pre-trained model
            result = model.predict(img_array)

            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f"Prediction: {prediction}")
