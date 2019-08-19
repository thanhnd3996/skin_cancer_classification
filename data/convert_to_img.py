import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

dataset = ""
des_dir = ""
for path in dataset:
    arr = np.load("")
    img = plt.imshow(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY))
    plt.savefig(os.path.join(des_dir, img))
