import os
import sys
import numpy as np
import base64
from tensorflow import keras
try:
    sys.path.append('/usr/local/lib/python3.7/site-packages')
except Exception as e:
    pass
import cv2

# Directory Paths
MODEL__DIR = os.path.join(os.getcwd(), "src/server/model_marzocco_detector.h5")


class ImagePredictor:
    def __init__(self):
        self.IMG_SIZE = 100
        self.test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)

    def predictImage(self, photo_path, model):
        x = []
        try:
            img_array = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
            x.append([new_array])
        except Exception as e:
            raise e

        pred = 0
        try:
            x = np.array(x).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
            pred = model.predict(self.test_datagen.flow(
                x, batch_size=1)[0])[0][0].item()
        except Exception as e:
            raise e
        return pred

    def predictImages(self, dir_path, model):
        err_count = 0
        preds = []
        for photo_reference in os.listdir(dir_path):
            try:
                photo_path = os.path.join(dir_path, photo_reference)
                pred = self.predictImage(photo_path, model)
                preds.append([pred, photo_reference])
            except Exception as e:
                if err_count > 3:
                    raise e
                else:
                    pass
        return preds

    # Helper Methods
    def saveImage(self, img_path, buffer):
        try:
            image_64_decode = base64.b64decode(buffer)
            image_result = open(img_path, "wb")
            image_result.write(image_64_decode)
        except Exception as e:
            raise Exception("Buffer sent not image compatible")

    def deleteImage(self, img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            raise Exception("Error deleting image")
