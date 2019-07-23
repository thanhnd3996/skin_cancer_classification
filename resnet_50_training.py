from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image

batch_size = 8
epochs = 1000

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
                                                    target_size=(224, 224), batch_size=batch_size)

valid_generator = val_datagen.flow_from_directory("./dataset/val_images/",
                                                  target_size=(224, 224), batch_size=batch_size)

# define model architecture
# load base_model
base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
last = base_model.output

x = Dense(2048, activation='relu')(last)
x = GlobalAveragePooling2D()(x)
preds = Dense(7, activation='softmax')(x)
model = Model(input=base_model.input, output=preds)
model.summary()

# Compile model
sgd = SGD(lr=0.1, clipnorm=5.)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Save the model with best weights
checkpointer = ModelCheckpoint('./checkpoint/resnet_50.h5',
                               verbose=1, save_best_only=True,
                               save_weights_only=True,
                               monitor='val_acc', mode="max")

# train model
model.fit_generator(train_generator, steps_per_epoch=8012 // batch_size, validation_data=valid_generator,
                    validation_steps=2003 // batch_size, epochs=epochs, verbose=1, callbacks=[checkpointer])
