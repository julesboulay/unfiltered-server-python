import tensorflow as tf
from tensorflow import keras
import pickle
import os

DATADIR = os.path.join(os.getcwd(), "marzocco_detector")
batch_size = 32; 

X = pickle.load(
    open(os.path.join(DATADIR, "images_data/images.pickle"), "rb"))
y = pickle.load(
    open(os.path.join(DATADIR, "images_data/present.pickle"), "rb"))
#X = X / 255.0

'''
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
model.fit(X, y, epochs=10)
'''
# define and fit the final model
l = tf.keras.layers
model = tf.keras.models.Sequential()
model.add(l.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100, 100, 1)))
model.add(l.MaxPooling2D(pool_size=(2, 2)))
model.add(l.Conv2D(64, (3, 3), activation='relu'))
model.add(l.MaxPooling2D(pool_size=(2, 2)))
model.add(l.Conv2D(128, (3, 3), activation='relu'))
model.add(l.MaxPooling2D(pool_size=(2, 2)))
model.add(l.Conv2D(128, (3, 3), activation='relu'))
model.add(l.MaxPooling2D(pool_size=(2, 2)))
model.add(l.Flatten())
model.add(l.Dropout(0.5))
model.add(l.Dense(512, activation='relu'))
model.add(l.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['acc'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow(X, y, batch_size=batch_size)
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=len(X) // batch_size, 
    epochs=64)

model.save(os.path.join(DATADIR, "model_marzocco_detector.h5"))
