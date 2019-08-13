from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image

# parameter
batch_size = 8
epochs = 1000
num_classes = 2
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
train_generator = train_datagen.flow_from_directory("../dataset_4/train_images/",
                                                    target_size=(299, 299), batch_size=batch_size)

valid_generator = val_datagen.flow_from_directory("../dataset_4/val_images/",
                                                  target_size=(299, 299), batch_size=batch_size)

# define model architecture
# load base_model
base_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
last = base_model.output

x = Dense(512, activation='relu')(last)
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
preds = Dense(num_classes, activation='softmax')(x)
model = Model(input=base_model.input, output=preds)
model.summary()

# Compile model
sgd = SGD(lr=0.1, clipnorm=5.)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Save the model with best weights
checkpointer = ModelCheckpoint('../checkpoint/inception_v3_task_4.h5',
                               verbose=1, save_best_only=True,
                               save_weights_only=True,
                               monitor='val_acc', mode="max")

# train model
model.fit_generator(train_generator, steps_per_epoch=638 // batch_size, validation_data=valid_generator,
                    validation_steps=121 // batch_size, epochs=epochs, verbose=1, callbacks=[checkpointer])
