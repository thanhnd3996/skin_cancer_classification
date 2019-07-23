from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image

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
train_generator = train_datagen.flow_from_directory("./dataset/train_images/",
                                                    target_size=(299, 299), batch_size=8)

valid_generator = val_datagen.flow_from_directory("./dataset/val_images/",
                                                  target_size=(299, 299), batch_size=8)

# define model architecture
# load base_model
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
last = base_model.output

x = Dense(2048, activation='relu')(last)
x = Dropout(0.4)(x)
x = GlobalAveragePooling2D()(x)
preds = Dense(7, activation='softmax')(x)
model = Model(input=base_model.input, output=preds)
model.summary()


# transfer learning
for layer in base_model.layers:
    layer.trainable = False

sgd = SGD(lr=0.1, clipnorm=5.)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=8012 // 8, validation_data=valid_generator,
					validation_steps=2003 // 8, epochs=20, verbose=1)


# fine tune
for layer in model.layers[:229]:
    layer.trainable = False
for layer in model.layers[229:]:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Save the model with best weights
checkpointer = ModelCheckpoint('./checkpoint/inception_v3_fine_tune.h5',
                               verbose=1, save_weights_only=True,
                               monitor='val_acc', mode="max")
# train model
model.fit_generator(train_generator, steps_per_epoch=8012 // 8, validation_data=valid_generator,
                    validation_steps=2003 // 8, epochs=150, verbose=1, callbacks=[checkpointer])
