import numpy as np
import os
import random
import pickle
import sys

sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2


DATADIR = os.path.join(os.getcwd(), "marzocco_detector")
IMG_SIZE = 100

MARZOCCO_TYPES = ["random", "fb80", "gb5", "linea", "strada"]
MARZOCCO_LABEL = [0       ,    1.0,   1.0,     1.0,      1.0]


def create_training_data():
    training_data = []
    _dir = os.path.join(DATADIR, "images/train")
    for _type_ in MARZOCCO_TYPES:
        path = os.path.join(_dir, _type_)
        label = MARZOCCO_LABEL[MARZOCCO_TYPES.index(_type_)]
        for img in os.listdir(path):
            try:
                new_path = os.path.join(path, img)
                img_array = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, label])
            except Exception as e:
                pass
    
    _dir = os.path.join(DATADIR, "images/test")
    for _type_ in MARZOCCO_TYPES:
        path = os.path.join(_dir, _type_)
        label = MARZOCCO_LABEL[MARZOCCO_TYPES.index(_type_)]
        for img in os.listdir(path):
            try:
                new_path = os.path.join(path, img)
                img_array = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, label])
            except Exception as e:
                pass

    return training_data


training_data = create_training_data()
random.shuffle(training_data)

marzocco_images = []
marzocco_labels = []
for img, label in training_data:
    marzocco_images.append(img)
    marzocco_labels.append(label)

marzocco_images = np.array(marzocco_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open(os.path.join(DATADIR, "images_data/images.pickle"), "wb")
pickle.dump(marzocco_images, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, "images_data/present.pickle"), "wb")
pickle.dump(marzocco_labels, pickle_out)
pickle_out.close()