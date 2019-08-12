import os
import numpy as np
from imutils import paths
import utils.read_img as r


def read_data(dataset):
    X = []
    y = []
    img_paths = sorted(list(paths.list_images(dataset)))
    for path in img_paths:
        img = r.read_image((299, 299), path)
        X.append(img)
        label = path.split(os.path.sep)[-2]
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y
