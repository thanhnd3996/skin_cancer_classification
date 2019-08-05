import os

import numpy as np
from keras import Model
from imutils import paths
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# parameter and file path
num_classes = 7
labels = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
trained_resnet_weights = "./checkpoint/resnet_50.h5"
trained_inception_weights = "./checkpoint/inception_v3.h5"
dataset = "./dataset/"
train_set = "./dataset/train_images/"
val_set = "./dataset/val_images"

# create model and load weight
print("Loading Resnet 50 ...")
resnet_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
resnet_last = resnet_model.output

resnet_fcn = GlobalAveragePooling2D()(resnet_last)
model_1 = Model(inputs=resnet_model.input, outputs=resnet_fcn)
model_1.load_weights(trained_resnet_weights, by_name=True)

print("Loading Inception V3")
inception_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
inception_last = inception_model.output

inception_fcn = GlobalAveragePooling2D()(inception_last)
model_2 = Model(inputs=inception_model.input, outputs=inception_fcn)
model_2.load_weights(trained_inception_weights, by_name=True)


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def feature_extractor(img_path):
    x_resnet = read_image((224, 224), img_path)
    x_inception = read_image((299, 299), img_path)

    ft_1 = model_1.predict(x_resnet)
    ft_reduce_1 = ft_1.squeeze()

    ft_2 = model_2.predict(x_inception)
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
