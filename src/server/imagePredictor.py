import os
import sys
import numpy as np
import zipfile
import base64
from tensorflow import keras
try:
    sys.path.append('/usr/local/lib/python3.7/site-packages')
except Exception as e:
    pass
import cv2

# Image Directories
SEARCH_DIR = os.path.join(os.getcwd(), "src/server/server_images/search")
HITS_DIR = os.path.join(os.getcwd(), "src/server/server_images/hits")


# SetUp the Model
MODEL__DIR = os.path.join(os.getcwd(), "src/server/model_marzocco_detector.h5")
zip_ref = zipfile.ZipFile("src/server/model_marzocco_detector.h5.zip", 'r')
zip_ref.extractall(os.getcwd())
zip_ref.close()

model = keras.models.load_model(MODEL__DIR)
model._make_predict_function()
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


class ImagePredictor:
    def __init__(self):
        self.IMG_SIZE = 100

    def predictImage(self, photo_reference, buffer):
        img_path = os.path.join(SEARCH_DIR, photo_reference + ".jpg")
        pred = 0
        try:
            self.__saveImage(img_path, buffer)
            x = self.__loadImageArray(os.getcwd(), photo_reference)
            pred = self.__modelPrediction(x)
            self.__deleteImage(img_path)
        except Exception as e:
            raise e

        return pred

    def predictImages(self, dir_path):
        preds = []
        try:
            x = self.__loadImagesArray(dir_path)
            preds = self.__modelPredictions(x)
        except Exception as e:
            raise e

        return preds

    def __modelPredictions(self, x):
        err_count = 0
        preds = []
        for single_x, photo_reference in x:
            try:
                pred = self.__modelPrediction(single_x)
                preds.append([pred, photo_reference])
            except Exception as e:
                if err_count < 3:
                    pass
                else:
                    raise e
        return preds

    def __modelPrediction(self, x):
        pred = 0
        try:
            new_x = np.array(x).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
            pred = model.predict(test_datagen.flow(
                new_x, batch_size=1)[0])[0][0].item()
        except Exception as e:
            print("ERROR HERE", str(e))
            raise Exception("Error calculating prediction")
        return pred

    def __loadImagesArray(self, dir_path):
        err_count = 0
        x = []
        for photo_referenece in os.listdir(dir_path):
            try:
                img_array = self.__loadImageArray(dir_path, photo_referenece)
                x.append([img_array, photo_referenece])
            except Exception as e:
                if err_count < 3:
                    pass
                else:
                    raise e
        return x

    def __loadImageArray(self, dir_path, photo_reference):
        x = []
        try:
            img_path = os.path.join(dir_path, photo_reference)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
            x.append([new_array])
        except Exception as e:
            raise Exception("Error loading image into array")
        return x

    # Helper Methods
    def __saveImage(self, img_path, buffer):
        try:
            image_64_decode = base64.b64decode(buffer)
            image_result = open(img_path, "wb")
            image_result.write(image_64_decode)
        except Exception as e:
            raise Exception("Buffer sent not image compatible")

    def __deleteImage(self, img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            raise Exception("Error deleting image")
