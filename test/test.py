import numpy as np
from keras import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Flatten, GlobalAveragePooling2D
from keras.preprocessing import image

# parameter and file path
num_classes = 7
labels = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# create model and load weight
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
last = base_model.output
layer = GlobalAveragePooling2D()(last)

model = Model(inputs=base_model.input, outputs=layer)
model.summary()

# load only validation images
X_train = []
y_train = []

img_path = "/home/ndt3996/PycharmProjects/skin_cancer_isic_2018/test/0.0_ISIC_0024331.jpg"

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
# print(features)
features_reduce = features.squeeze()
X_train.append(features_reduce)

# generate one hot vector for label
targets = np.zeros(num_classes)
label = 'BCC'
for i in labels:
    if i == label:
        targets[labels.index(i)] = 1
y_train.append(targets)

X = np.array(X_train)
y = np.array(y_train, np.uint8)
