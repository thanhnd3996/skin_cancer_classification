import os

import numpy as np
from imutils import paths
from keras import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image

# parameter and file path
num_classes = 7
labels = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
trained_resnet_weights = "./checkpoint/resnet_50.h5"
dataset = "./dataset/"
val_set = "./dataset/val_images/"
test_set = "./dataset/test_images"

# create model and load weight
base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
last = base_model.output
layer = GlobalAveragePooling2D()(last)
model = Model(inputs=base_model.input, outputs=layer)
model.summary()
model.load_weights(trained_resnet_weights)

# load only validation images
X_train = []
y_train = []

image_paths = sorted(list(paths.list_images(val_set)))
for img_path in image_paths:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_reduce = features.squeeze()
    X_train.append(features_reduce)

    # generate one hot vector for label
    targets = np.zeros(num_classes)
    label = img_path.split(os.path.sep)[-2]
    for i in labels:
        if i == label:
            targets[labels.index(i)] = 1
    y_train.append(targets)

X = np.array(X_train)
y = np.array(y_train, np.uint8)

X_test = []
image_paths = sorted(list(paths.list_images(test_set)))
for img_path in image_paths:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_reduce = features.squeeze()
    X_train.append(features_reduce)
