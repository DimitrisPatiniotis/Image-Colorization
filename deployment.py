import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
import skimage
from skimage import io, color, transform
from tensorflow import keras


def get_LAB(image_l, image_ab):
    image_l = image_l.reshape((224, 224, 1))
    image_lab = np.concatenate((image_l, image_ab), axis=2)
    image_lab = image_lab.astype("uint8")
    image_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    image_rgb = Image.fromarray(image_rgb)
    return image_rgb


def to_rgb_image(l, ab):
    shape = (l.shape[0], l.shape[1], 3)
    img = np.zeros(shape)
    img[:, :, 0] = l[:, :, 0]
    img[:, :, 1:] = ab
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def get_pred(model, image_grayscale):
    input_image = np.repeat(image_grayscale[..., np.newaxis], 3, -1)
    input_image = input_image.reshape((1, 224, 224, 3))
    input_image = (input_image.astype('float32') - 127.5) / 127.5
    prediction = model.predict(input_image)
    pred = (prediction[0].astype('float32') * 127.5) + 127.5

    return pred


def to3channels(gray):
    shape = (gray.shape[0], gray.shape[1], 3)
    zeros = np.zeros(shape)
    zeros[:, :, 0] = gray
    zeros[:, :, 1] = gray
    zeros[:, :, 2] = gray
    return zeros


def l_channel_from_image(img_path):
    img = io.imread(img_path)
    img = transform.resize(img, (224, 224))
    gray = color.rgb2gray(img)
    gray = to3channels(gray)
    lgray = color.rgb2lab(gray, illuminant='D50')[:, :, 0]

    return lgray

st.title('Image Colorization demo')

uploaded_image = st.file_uploader(
        label='Pick a grayscale image to colorize', type=['png', 'jpg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    # Save image that's saved in RAM
    image.save("temp.png")
    grayscale_image = l_channel_from_image("temp.png")
    try:
        model = keras.models.load_model('models/auto_encoder_v2.0.model')
    except:
        model = keras.models.load_model('models\auto_encoder_v2.0.model')

    pred = get_pred(model, grayscale_image)
    image = get_LAB(grayscale_image, pred)
    new_image = Image.new('RGB', (448, 224))
    gray_image = Image.fromarray(grayscale_image)
    new_image.paste(gray_image, (0,0))
    new_image.paste(image, (224, 0))
    st.image(new_image)
    os.remove('temp.png')