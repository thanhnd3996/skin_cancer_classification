from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

# parameters
batch_size = 8
epochs = 150

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
args = parser.parse_args()

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args.dataset)))
random.seed(42)
random.shuffle(image_paths)

# loop over the input images
for image_path in image_paths:
    print(image_path)
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (299, 299))
    image = np.reshape(image, (299, 299, 3))
    print(image)
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    label = np.reshape(label, -1)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)
print(data)
print(labels)

# partition the data into training and validation
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# data generator
train_aug = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               vertical_flip=True)

val_aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# init the model and optimizer
base_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
last = base_model.output

x = Dense(512, activation='relu')(last)
x = GlobalAveragePooling2D()(x)
preds = Dense(7, activation='softmax')(x)
model = Model(input=base_model.input, output=preds)
model.summary()

# train the network
sgd = SGD(lr=0.1, clipnorm=5.)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Save the model with best weights
checkpointer = ModelCheckpoint('./checkpoint/inception_v3_1.h5',
                               verbose=1, save_best_only=True,
                               monitor='val_acc', mode="max")

print("[INFO] training network...")
model.fit_generator(train_aug.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=epochs,
                    validation_steps=len(x_test) // batch_size,
                    callbacks=[checkpointer])
