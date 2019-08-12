import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
