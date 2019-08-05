import numpy as np
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

# parameter and file path
num_classes = 7
# labels = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# create model and load weight
base_model_1 = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
layer_1 = GlobalAveragePooling2D()(base_model_1.output)

base_model_2 = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
layer_2 = GlobalAveragePooling2D()(base_model_2.output)

model_1 = Model(inputs=base_model_1.input, outputs=layer_1)
model_2 = Model(inputs=base_model_2.input, outputs=layer_2)

# load only validation images
X_train = []
y_train = []

img_path = "/home/ndt3996/PycharmProjects/skin_cancer_isic_2018/test/0.0_ISIC_0024331.jpg"


def read_image(target_size):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


x_resnet = read_image((224, 224))
x_inception = read_image((299, 299))

features_1 = model_1.predict(x_resnet)
features_2 = model_2.predict(x_inception)
features = np.add(features_1, features_2)
print(features.shape)
print(type(features))
# features_reduce = features.squeeze()
# X_train.append(features_reduce)
#
# # generate one hot vector for label
# # targets = np.zeros(num_classes)
# label = 'BCC'
# # for i in labels:
# #     if i == label:
# #         targets[labels.index(i)] = 1
# y_train.append(label)
#
# X = np.array(X_train)
# y = np.array(y_train)
# # print(X.shape)
# # print(y.shape)
# label_encoder = LabelEncoder()
# label_encoder = label_encoder.fit(y_train)
# y_train = label_encoder.transform(y_train)
# print(y_train)
