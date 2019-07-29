import os

import numpy as np
from imutils import paths
from keras import Model
from sklearn.ensemble import RandomForestClassifier
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
last = base_model.output

layer = Dense(2048, activation='relu')(last)
layer = GlobalAveragePooling2D()(layer)
model = Model(inputs=base_model.input, outputs=layer)
model.load_weights(trained_resnet_weights)

print("Loading Inception V3")
inception_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
inception_last = inception_model.output

inception_layer = Dense(2048, activation='relu')(inception_last)
inception_layer = GlobalAveragePooling2D()(inception_layer)
model_2 = Model(inputs=inception_model.input, outputs=inception_layer)
model_2.load_weights(trained_inception_weights)
model_2.summary()


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# load only validation images
X_train = []
y_train = []

train_paths = sorted(list(paths.list_images(train_set)))
for train_path in train_paths:
    x_resnet = read_image((224, 224), train_path)
    x_inception = read_image((299, 299), train_path)

    ft_1 = model.predict(x_resnet)
    ft_2 = model_2.predict(x_inception)
    features_reduce_1 = ft_1.squeeze()
    features_reduce_2 = ft_2.squeeze()
    ft_vector = np.add(features_reduce_1, features_reduce_2)
    X_train.append(ft_vector)

    # generate one hot vector for label
    label = train_path.split(os.path.sep)[-2]
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train, np)
train_lb_encoder = LabelEncoder()
train_lb_encoder = train_lb_encoder.fit(y_train)
y_train = train_lb_encoder.transform(y_train)

X_test = []
y_test = []
val_paths = sorted(list(paths.list_images(val_set)))

for val_path in val_paths:
    x_resnet = read_image((224, 224), val_path)
    x_inception = read_image((299, 299), val_path)

    ft_1 = model.predict(x_resnet)
    ft_2 = model_2.predict(x_inception)
    features_reduce_1 = ft_1.squeeze()
    features_reduce_2 = ft_2.squeeze()
    ft_vector = np.add(features_reduce_1, features_reduce_2)
    X_test.append(ft_vector)

    label = val_path.split(os.path.sep)[-2]
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)
val_lb_encoder = LabelEncoder()
val_lb_encoder = val_lb_encoder.fit(y_test)
y_test = val_lb_encoder.transform(y_test)

print('Training and making predictions')
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))
