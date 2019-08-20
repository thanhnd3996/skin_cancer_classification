import matplotlib
import numpy as np
from imutils import paths
from matplotlib import image

dataset = "../preprocess/"
des_dir = "../new_dataset/train/"

img_paths = sorted(list(paths.list_images(dataset)))
for img_path in img_paths:
    diagnosis = img_path.split("/")[-2]
    img_name = img_path.split("/")[-1]
    img_name = img_name.split(".")[-2]
    img_name = img_name.split
    arr = np.load(img_path)
    img_name = img_name + ".jpg"
    matplotlib.image.imsave(des_dir + diagnosis + img_name, arr)
