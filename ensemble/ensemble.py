import os

import numpy as np
from keras import Model
from imutils import paths
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# parameter and file path
num_classes = 7
labels = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
dataset = "../dataset/"
train_set = "../dataset/train_images/"
val_set = "../dataset/val_images/"
test_set = "../dataset/test_images/"
task_1_weight = "../checkpoint/inception_v3_dataset_1.h5"
task_2_weight = "../checkpoint/inception_v3_task_2.h5"
task_3_weight = "../checkpoint/inception_v3_task_3.h5"
task_4_weight = "../checkpoint/inception_v3_task_4.h5"
# create model
print("Loading Inception V3")
inception_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
inception_last = inception_model.output
# for task 1 and task 2
# inception_fcn = Dense(2048, activation='relu')(inception_last)
# inception_fcn = GlobalAveragePooling2D()(inception_fcn)
inception_fcn = GlobalAveragePooling2D()(inception_last)
model_1 = Model(inputs=inception_model.input, outputs=inception_fcn)
model_1.load_weights(task_1_weight, by_name=True)
model_2 = Model(inputs=inception_model.input, outputs=inception_fcn)
model_2.load_weights(task_2_weight, by_name=True)

# for task 3 and task 4
# fcn2 = Dense(512, activation='relu')(inception_last)
# fcn2 = Dropout(0.5)(fcn2)
# fcn2 = GlobalAveragePooling2D()(fcn2)
fcn2 = GlobalAveragePooling2D()(inception_last)
model_3 = Model(inputs=inception_model.input, outputs=fcn2)
model_3.load_weights(task_3_weight, by_name=True)
model_4 = Model(inputs=inception_model.input, outputs=fcn2)
model_4.load_weights(task_4_weight, by_name=True)


def read_image(target_size, img_path):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def feature_extractor(img_path):
    x = read_image((299, 299), img_path)

    ft_1 = model_1.predict(x)
    ft_2 = model_2.predict(x)
    ft_3 = model_3.predict(x)
    ft_4 = model_4.predict(x)

    ft_reduce_1 = ft_1.squeeze()
    ft_reduce_2 = ft_2.squeeze()
    ft_reduce_3 = ft_3.squeeze()
    ft_reduce_4 = ft_4.squeeze()

    ft_vector = np.concatenate((ft_reduce_1, ft_reduce_2, ft_reduce_3, ft_reduce_4), axis=0)

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
    test_lb_encoder = LabelEncoder()
    test_lb_encoder = test_lb_encoder.fit(y_test)
    y_test = test_lb_encoder.transform(y_test)
    np.save('X_val.npy', X_test)
    np.save('y_val.npy', y_test)


if __name__ == '__main__':
    create_train_data()
    create_test_data()
