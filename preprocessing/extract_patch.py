import os
import cv2
from imutils import paths
from sklearn.feature_extraction import image as patch_ex
from matplotlib import image

PATCH_SIZE = 299

train = "../preprocess/train_images/"
val = "../preprocess/val_images/"
new_train = "../new_dataset/train_images/"
new_val = "../new_dataset/val_images/"


def extract_path(dataset, new_dataset, num_per_patch=None):
    img_paths = sorted(list(paths.list_images(dataset)))
    for img_path in img_paths:

        diag = img_path.split("/")[-2]
        if diag == 'AKIEC':
            num_per_patch = 20
        elif diag == 'BCC':
            num_per_patch = 13
        elif diag == 'BKL':
            num_per_patch = 6
        elif diag == 'DF':
            num_per_patch = 60
        elif diag == 'MEL':
            num_per_patch = 6
        elif diag == 'NV':
            num_per_patch = 1
        elif diag == 'VASC':
            num_per_patch = 50
        img_name = img_path.split("/")[-1]
        img_name = img_name.split(".")[-2]
        new_path = new_dataset + diag + "/"

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        patches = patch_ex.extract_patches_2d(img, (PATCH_SIZE, PATCH_SIZE), num_per_patch)
        i = 0

        for patch in patches:
            i += 1
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            image.imsave(new_path + img_name + "_" + str(i) + ".jpg", patch)


if __name__ == '__main__':
    extract_path(train, new_train)
    extract_path(val, new_val)
