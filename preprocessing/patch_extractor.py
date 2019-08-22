import shutil
import sys
from os import makedirs
from os.path import basename, join, exists

import cv2
import numpy as np
from sklearn.feature_extraction import image

PATCH_SIZE = 299
NUM_PER_PATCH = 20

INPUT_DIR = 'preprocess/test'
PREPROCESS_DIR = 'new_dataset//'
CLASSES = ['test']


def recursive_glob(root_dir, file_template="*.jpg"):
    """Traverse directory recursively. Starting with Python version 3.5, the glob module supports the "**" directive"""

    if sys.version_info[0] * 10 + sys.version_info[1] < 35:
        import fnmatch
        import os
        matches = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, file_template):
                matches.append(os.path.join(root, filename))
        return matches
    else:
        import glob
        return glob.glob(root_dir + "/**/" + file_template, recursive=True)


def extract_patch(classes_name):
    for patch_type in classes_name:
        if patch_type == 'test':
            input_files = recursive_glob(INPUT_DIR)
            if not exists(PREPROCESS_DIR):
                makedirs(PREPROCESS_DIR)
            else:
                shutil.rmtree(PREPROCESS_DIR)
                makedirs(PREPROCESS_DIR)
        else:
            input_files = recursive_glob(join(INPUT_DIR, patch_type))
            if not exists(join(PREPROCESS_DIR, patch_type)):
                makedirs(join(PREPROCESS_DIR, patch_type))
            else:
                shutil.rmtree(join(PREPROCESS_DIR, patch_type))
                makedirs(join(PREPROCESS_DIR, patch_type))

        for f in input_files:
            if len(f.split("/")) > 2:
                class_name = f.split("/")[2]
            else:
                class_name = ""

            s = list(basename(f))

            if patch_type == 'test':
                SAVE_DIR = PREPROCESS_DIR
            else:
                SAVE_DIR = join(PREPROCESS_DIR, class_name)
            print('-----------Patch Extractor-------------')
            print(s)

            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            patches = image.extract_patches_2d(img, (PATCH_SIZE, PATCH_SIZE), NUM_PER_PATCH)

            for index, patch in enumerate(patches):

                filename_origin = s[:len(s) - 4]
                patch_rt = np.rot90(patch, np.random.randint(low=0, high=4))

                print(join(SAVE_DIR, filename_origin))
                np.save(join(SAVE_DIR, filename_origin), patch_rt)


if __name__ == "__main__":
    extract_patch(CLASSES)
