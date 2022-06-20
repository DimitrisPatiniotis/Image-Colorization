import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
import skimage
import cv2
from PIL import Image

def display(img):
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(img)
    plt.show()

def get_LAB(image_l, image_ab):
    image_l = image_l.reshape((224, 224, 1))
    image_lab = np.concatenate((image_l, image_ab), axis=2)
    image_lab = image_lab.astype("uint8")
    image_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    image_rgb = Image.fromarray(image_rgb)
    return image_rgb

def to_rgb_image(l, ab):
    shape = (l.shape[0],l.shape[1],3)
    img = np.zeros(shape)
    img[:,:,0] = l[:,:,0]
    img[:,:,1:]= ab
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
    img = skimage.transform.resize(img,(224,224))
    gray = color.rgb2gray(img)
    gray = to3channels(gray)
    lgray = color.rgb2lab(gray, illuminant='D50')[:, :, 0]
    return lgray

if __name__ == '__main__':
    display(l_channel_from_image('../images/v0.0/result.png'))