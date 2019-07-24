from keras.applications.resnet50 import ResNet50

base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
last = base_model.output
x = Flatten()(last)
model = Model(inputs=base_model, outputs=x)