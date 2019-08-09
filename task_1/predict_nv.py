from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

test_dir = "../dataset_1/test_images/"
checkpoint_path_2 = '../checkpoint/inception_v3_dataset_1.h5'


def evaluate():
    test_gen_args = dict(horizontal_flip=True, vertical_flip=True)
    test_data_gen = image.ImageDataGenerator(**test_gen_args)
    test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(299, 299),
        batch_size=8,
        class_mode="categorical"
    )
    file_names = test_generator.filenames
    nb_samples = len(file_names)

    # load base_model
    base_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
    last = base_model.output

    x = Dense(2048, activation='relu')(last)
    x = GlobalAveragePooling2D()(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=preds)

    # evaluate loaded model on test data
    sgd = SGD(lr=0.1, clipnorm=5.)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    score = model.predict_generator(test_generator, steps=nb_samples)

    for i in range(0, nb_samples):
        idx = np.argmax(score[i])
        print(idx, score[i][idx])

    y_pred = np.argmax(score, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    print(cm.diagonal())
    # plot_confusion_matrix(cm, idx, normalize=True, title="Normalize confusion matrix")
    # plt.show()


#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     fmt = ".2f" if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalaligment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#         plt.tight_layout()
#         plt.ylabel("True label")
#         plt.xlabel("Predicted label")
#

if __name__ == '__main__':
    evaluate()
