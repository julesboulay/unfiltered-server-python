import os
import sys
import json
import zipfile
import numpy as np
from tensorflow import keras
try:
    sys.path.append('/usr/local/lib/python3.7/site-packages')
except Exception as e:
    pass
import cv2

MODEL__DIR = os.path.join(os.getcwd(), "src/server/model_marzocco_detector.h5")
IMG_SIZE = 100

# SetUp the Model
model = keras.models.load_model(MODEL__DIR)
model._make_predict_function()
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

photo_path = os.path.join(os.getcwd(), "src/server/try.jpg")
x = []
try:
    img_array = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    x.append([new_array])
except Exception as e:
    print(str(e))

pred = 0
try:
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(test_datagen.flow(
        x, batch_size=1)[0])[0][0].item()
except Exception as e:
    print(str(e))

print(pred)
