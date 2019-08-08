import os

import numpy as np
from imutils import paths
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

# parameter and file path
num_classes = 7
labels = ['NV', 'Malignant', 'cancer_2', 'benign']
trained_nv_classify = "../checkpoint/inception_v3_dataset_1.h5"
trained_malignant_classify = "../checkpoint/inception_v3_task2.h5"

train_set = "../dataset_task_1_2/"
val_set = "../dataset_task_1_2/"

# create model and load weight
print("Loading model for task 1")

print("Loading Inception V3")
inception_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
inception_last = inception_model.output

inception_fcn = GlobalAveragePooling2D()(inception_last)

model_1 = Model(inputs=inception_model.input, outputs=inception_fcn)
model_2 = Model(inputs=inception_model.input, outputs=inception_fcn)

model_1.load_weights(trained_nv_classify, by_name=True)
model_2.load_weights(trained_malignant_classify, by_name=True)


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def feature_extractor(img_path):
    x_1 = read_image((299, 299), img_path)
    x_2 = read_image((299, 299), img_path)

    ft_1 = model_1.predict(x_1)
    ft_reduce_1 = ft_1.squeeze()

    ft_2 = model_2.predict(x_2)
    ft_reduce_2 = ft_2.squeeze()

    ft_vector = np.concatenate((ft_reduce_1, ft_reduce_2), axis=0)

    return ft_vector


def create_train_data():
    X_train = []
    y_train = []
    train_paths = sorted(list(paths.list_images(train_set)))
    for train_path in train_paths:
        train_ft = feature_extractor(train_path)
        X_train.append(train_ft)
        label = train_path.split(os.path.sep)[-2]
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train, np)
    train_lb_encoder = LabelEncoder()
    train_lb_encoder = train_lb_encoder.fit(y_train)
    y_train = train_lb_encoder.transform(y_train)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)


def create_val_data():
    X_val = []
    y_val = []
    val_paths = sorted(list(paths.list_images(val_set)))
    for val_path in val_paths:
        val_ft = feature_extractor(val_path)
        X_val.append(val_ft)
        label = val_path.split(os.path.sep)[-2]
        y_val.append(label)

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    val_lb_encoder = LabelEncoder()
    val_lb_encoder = val_lb_encoder.fit(y_val)
    y_val = val_lb_encoder.transform(y_val)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)


if __name__ == '__main__':
    create_train_data()
    create_val_data()
