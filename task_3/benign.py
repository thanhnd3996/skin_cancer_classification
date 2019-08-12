from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from imblearn.over_sampling import SMOTE
from utils import read_data as r

# parameter
batch_size = 8
epochs = 1000
num_classes = 3
train_set = "../dataset_3/train_images/"
val_set = "../dataset_3/val_images"
test_set = "../dataset_3/test_images"
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
X_train, y_train = r.read_data(train_set)
X_val, y_val = r.read_data(val_set)

# Up-sampling train image
sm = SMOTE(random_state=12, ratio=1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

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
checkpointer = ModelCheckpoint('../checkpoint/inception_v3_task_3.h5',
                               verbose=1, save_best_only=True,
                               save_weights_only=True,
                               monitor='val_acc', mode="max")

# train model
model.fit_generator(train_datagen.flow(X_train_res, y_train_res, batch_size=batch_size),
                    steps_per_epoch=len(X_train_res) // batch_size, validation_data=(X_val, y_val),
                    epochs=epochs, verbose=1, callbacks=[checkpointer])
