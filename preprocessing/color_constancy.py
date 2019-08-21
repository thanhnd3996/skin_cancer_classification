import os
import cv2
import numpy as np
from matplotlib import image

# path
train_dataset = "../dataset/train_images/"
val_dataset = "../dataset/val_images/"
new_train = "../preprocessing/train_images/"
new_val = "../preprocessing/val_images/"


def color_constancy(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)


def preprocess(dataset, des_dir):
    for root, dirs, files in os.walk(dataset):
        for file in files:
            path = os.path.join(root, file)
            new_path = des_dir + path.split("/")[-2] + "/"
            im = cv2.imread(path)
            im_arr = color_constancy(im)
            rgb_im = cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            image.imsave(new_path + file, rgb_im)


if __name__ == '__main__':
    preprocess(train_dataset, new_train)
    preprocess(val_dataset, new_val)
