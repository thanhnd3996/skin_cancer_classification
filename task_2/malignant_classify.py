from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras import Sequential
from keras.optimizers import SGD
from keras.preprocessing import image

# parameter
batch_size = 8
epochs = 1000
num_classes = 3
# data pre-processing
train_gen_args = dict(rotation_range=30,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      shear_range=0.2,
                      zoom_range=0.2,
                      horizontal_flip=True,
                      vertical_flip=True)

val_gen_args = dict(horizontal_flip=True, vertical_flip=True)

train_datagen = image.ImageDataGenerator(**train_gen_args)
val_datagen = image.ImageDataGenerator(**val_gen_args)

# load image
train_generator = train_datagen.flow_from_directory("./dataset_2/train_images/",
                                                    target_size=(299, 299), batch_size=batch_size)

valid_generator = val_datagen.flow_from_directory("./dataset_2/val_images/",
                                                  target_size=(299, 299), batch_size=batch_size)

# define model architecture
# load base_model
benchmark = Sequential()
benchmark.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(299, 299, 3)))
benchmark.add(MaxPooling2D(pool_size=2, padding='same'))
benchmark.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
benchmark.add(MaxPooling2D(pool_size=2, padding='same'))
benchmark.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
benchmark.add(MaxPooling2D(pool_size=2, padding='same'))
benchmark.add(Dropout(0.3))
benchmark.add(Flatten())
benchmark.add(Dense(512, activation='relu'))
benchmark.add(Dropout(0.5))
benchmark.add(Dense(3, activation='softmax'))
benchmark.summary()

# Compile model
sgd = SGD(lr=0.1, clipnorm=5.)

benchmark.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Save the model with best weights
checkpointer = ModelCheckpoint('../checkpoint/inception_v3_task_2.h5',
                               verbose=1, save_best_only=True,
                               save_weights_only=True,
                               monitor='val_acc', mode="max")

# train model
benchmark.fit_generator(train_generator, steps_per_epoch=8012 // batch_size, validation_data=valid_generator,
                        validation_steps=2003 // batch_size, epochs=epochs, verbose=1, callbacks=[checkpointer])
