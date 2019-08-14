import os

import numpy as np
from imutils import paths
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# parameter and file path
num_classes = 2
labels = ['nv', 'non_nv']
trained_task2_model = "../checkpoint/inception_v3_dataset_1.h5"
train_set = "../dataset_1/train_images/"
val_set = "../dataset_1/val_images"
test_set = "../dataset_1/test_images"

# create model and load weight

print("Loading Inception V3")
inception_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
inception_last = inception_model.output

inception_fcn = Dense(2048, activation='relu')(inception_last)
inception_fcn = GlobalAveragePooling2D()(inception_fcn)
model_1 = Model(inputs=inception_model.input, outputs=inception_fcn)
model_1.load_weights(trained_task2_model, by_name=True)


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def feature_extractor(img_path):
    x_1 = read_image((299, 299), img_path)

    ft_1 = model_1.predict(x_1)
    ft_vector = ft_1.squeeze()

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
    y_train = np.array(y_train)
    train_lb_encoder = LabelEncoder()
    train_lb_encoder = train_lb_encoder.fit(y_train)
    y_train = train_lb_encoder.transform(y_train)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)


def create_test_data():
    X_test = []
    y_test = []
    test_paths = sorted(list(paths.list_images(test_set)))
    for test_path in test_paths:
        test_ft = feature_extractor(test_path)
        X_test.append(test_ft)
        label = test_path.split(os.path.sep)[-2]
        y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    val_lb_encoder = LabelEncoder()
    val_lb_encoder = val_lb_encoder.fit(y_test)
    y_test = val_lb_encoder.transform(y_test)
    return X_test, y_test


if __name__ == '__main__':
    create_train_data()
    create_test_data()
