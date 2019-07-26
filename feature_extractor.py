import os

import numpy as np
from imutils import paths
from keras import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing import image
import xgboost as xgb
from sklearn.metrics import accuracy_score

# parameter and file path
num_classes = 7
labels = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
trained_resnet_weights = "./checkpoint/resnet_50.h5"
dataset = "./dataset/"
train_set = "./dataset/train_images/"
val_set = "./dataset/val_images"

# create model and load weight
base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
last = base_model.output
layer = Dense(2048, activation='relu')(last)
layer = Dropout(0.4)(layer)
layer = GlobalAveragePooling2D()(layer)
model = Model(inputs=base_model.input, outputs=layer)
model.summary()
model.load_weights(trained_resnet_weights)

# load only validation images
X_train = []
y_train = []

train_paths = sorted(list(paths.list_images(train_set)))
for train_path in train_paths:
    img = image.load_img(train_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_reduce = features.squeeze()
    X_train.append(features_reduce)

    # generate one hot vector for label
    targets = np.zeros(num_classes)
    label = train_path.split(os.path.sep)[-2]
    for i in labels:
        if i == label:
            targets[labels.index(i)] = 1
    y_train.append(targets)

X_train = np.array(X_train)
y_train = np.array(y_train, np.uint8)

X_test = []
y_test = []
val_paths = sorted(list(paths.list_images(val_set)))
for val_path in val_paths:
    img = image.load_img(val_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_reduce = features.squeeze()
    X_test.append(features_reduce)

    targets = np.zeros(num_classes)
    label = val_path.split(os.path.sep)[-2]
    for i in labels:
        if i == label:
            targets[labels.index(i)] = 1
    y_test.append(targets)

X_test = np.array(X_test)
y_test = np.array(y_test, np.uint8)

print('Training and making predictions')
clf = xgb.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=200)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))
