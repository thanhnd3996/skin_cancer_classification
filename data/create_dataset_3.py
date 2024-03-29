"""
Dataset_3 is dataset use for 3 benign class classification.
"""

import os
import pandas as pd
import shutil


def format_dir(train_val_dir, target_dir, name_csv):
    # format dir into keras on-the-fly image generator format

    # train and val data frames
    df = pd.read_csv(name_csv)

    # loop over image
    for ix, (image, diagnosis) in df.iterrows():
        # get names

        if diagnosis == 'BKL':
            original_img_path = "%s%s" % (train_val_dir, image + '.jpg')
            new_img_dir = "%s%s/" % (target_dir, 'BKL')
            new_img_path = "%s%s" % (new_img_dir, image + '.jpg')
            # create category dir
            if not os.path.exists(new_img_dir):
                os.makedirs(new_img_dir)
            # copy image
            shutil.copy(original_img_path, new_img_path)
        elif diagnosis == 'DF':
            original_img_path = "%s%s" % (train_val_dir, image + '.jpg')
            new_img_dir = "%s%s/" % (target_dir, 'DF')
            new_img_path = "%s%s" % (new_img_dir, image + '.jpg')
            # create category dir
            if not os.path.exists(new_img_dir):
                os.makedirs(new_img_dir)
            # copy image
            shutil.copy(original_img_path, new_img_path)
        elif diagnosis == 'VASC':
            original_img_path = "%s%s" % (train_val_dir, image + '.jpg')
            new_img_dir = "%s%s/" % (target_dir, 'VASC')
            new_img_path = "%s%s" % (new_img_dir, image + '.jpg')
            # create category dir
            if not os.path.exists(new_img_dir):
                os.makedirs(new_img_dir)
            # copy image
            shutil.copy(original_img_path, new_img_path)
        else:
            pass


if __name__ == '__main__':
    format_dir(train_val_dir="../ISIC2018_input/",
               target_dir="../dataset_3/train_images/",
               name_csv="train.csv")
    format_dir(train_val_dir="../ISIC2018_input/",
               target_dir="../dataset_3/val_images/",
               name_csv="val.csv")
    format_dir(train_val_dir="../ISIC2018_input/",
               target_dir="../dataset_3/test_images/",
               name_csv="test.csv")
