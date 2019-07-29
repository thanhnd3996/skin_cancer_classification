import os
import numpy as np
import xgboost as xgb
from keras import Model
from imutils import paths
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# file path
train_set = "./dataset/train_images/"
val_set = "./dataset/val_images"

# create model
resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 244, 3))
l1 = GlobalAveragePooling2D()(resnet_model.output)
inception_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
l2 = GlobalAveragePooling2D()(inception_model.output)

model1 = Model(inputs=resnet_model.input, outputs=l1)
model2 = Model(inputs=inception_model.input, outputs=l2)


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


X_train = []
y_train = []
X_test = []
y_test = []

train_paths = sorted(list(paths.list_images(train_set)))
val_paths = sorted(list(paths.list_images(val_set)))

for train_path in train_paths:
    x_1 = read_image((224, 224), train_path)
    x_2 = read_image((299, 299), train_path)

    ft_1 = model1.predict(x_1)
    ft_2 = model2.predict(x_2)
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

for val_path in val_paths:
    x_1 = read_image((224, 224), val_path)
    x_2 = read_image((299, 299), val_path)

    ft_1 = model1.predict(x_1)
    ft_2 = model2.predict(x_2)
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
clf = xgb.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=200)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))
