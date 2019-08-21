import matplotlib
import numpy as np
from matplotlib import image
import os

dataset = "../preprocess/val/"
des_dir = "../new_dataset/val_images/"

for root, dirs, files in os.walk(dataset):

    for file in files:
        p = os.path.join(root, file)
        diagnosis = p.split("/")[-2] + "/"
        img_name = p.split("/")[-1]
        img_name = img_name.split(".")[-2]
        arr = np.load(p)
        img_name = img_name + ".jpg"
        if not os.path.exists(des_dir + diagnosis):
            os.makedirs(des_dir + diagnosis)
        matplotlib.image.imsave(des_dir + diagnosis + img_name, arr)
