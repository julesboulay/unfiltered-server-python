import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow import keras
import pickle
import os

DATADIR = os.path.join(os.getcwd(), "marzocco_detector")

marzocco_images = pickle.load(
    open(os.path.join(DATADIR, "images_data/images.pickle"), "rb"))
marzocco_labels = pickle.load(
    open(os.path.join(DATADIR, "images_data/present.pickle"), "rb"))
marzocco_images = marzocco_images / 255.0

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adadelta(),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

model.fit(marzocco_images, marzocco_labels, epochs=10)

model.save(os.path.join(DATADIR, "model_marzocco_detector.h5"))
