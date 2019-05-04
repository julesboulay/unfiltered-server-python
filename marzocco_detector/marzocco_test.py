import numpy as np
from tensorflow import keras
import os
import random
import sys

sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2


DATADIR = os.path.join(os.getcwd(), "marzocco_detector")
IMG_SIZE = 100

MARZOCCO_TYPES = ["random", "fb80", "gb5", "linea", "strada"]
MARZOCCO_LABEL = [       0,    1.0,   1.0,     1.0,      1.0]


def create_testing_data():
    testing_data = []
    _dir = os.path.join(DATADIR, "images/test")
    for _type_ in MARZOCCO_TYPES:
        path = os.path.join(_dir, _type_)
        label = MARZOCCO_LABEL[MARZOCCO_TYPES.index(_type_)]
        for img in os.listdir(path):
            try:
                new_path = os.path.join(path, img)
                img_array = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, label])
            except Exception as e:
                pass

    return testing_data

def create_predict_photo():
    x = []
    new_path = os.path.join(os.getcwd(), "server_image/photo.jpg")
    #for i in [1, 2]:
    try:
        img_array = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        x.append([new_array])
    except Exception as e:
        pass

    return x

x = create_predict_photo()
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

training_data = create_testing_data()
random.shuffle(training_data)

X = []
y = []
for img, label in training_data:
    X.append(img)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Load Model
model = keras.models.load_model(os.path.join(
    DATADIR, "model_marzocco_detector.h5"))
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# 0. Prepare Batch
'''
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    print(pred)
'''
pred = model.predict(test_datagen.flow(x, batch_size=1)[0])
print(pred[0][0])

'''
# 1. Test Individual Images
predictions = model.predict_proba(marzocco_images)
print("Individual Prediction: ", predictions[0])

# 2. Evaluate Accuracy
test_loss, test_acc = model.evaluate(marzocco_images, marzocco_labels)
print("Test accuracy:", test_acc)
'''