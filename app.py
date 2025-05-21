import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

# Load your trained MobileNetV2 model
model = tf.keras.models.load_model('pcos_mobilenetv2_model.h5')  # Update path if needed

# Grad-CAM helper functions
def get_img_array(image, size):
    image = image.resize(size)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image, heatmap, alpha=0.4):
    image = np.array(image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed

# Streamlit UI
st.title("PCOS Detection from Ultrasound")
st.write("Upload an ultrasound image to predict if the patient has PCOS and visualize the region of interest using Grad-CAM.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess and predict
    img_array = get_img_array(image, size=(224, 224))
    prediction = model.predict(img_array)[0][0]

    label = "PCOS Detected" if prediction > 0.5 else "Normal"
    st.subheader(f"Prediction: {label} (Confidence: {prediction:.2f})")

    # Grad-CAM visualization
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1')
    gradcam_image = display_gradcam(np.array(image), heatmap)

    st.subheader("Grad-CAM Visualization")
    st.image(gradcam_image, caption="Important Regions for Prediction",use_container_width=True)
