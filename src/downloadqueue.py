import os
import time
import requests
import shutil
import queue
from tensorflow import keras
from threading import Thread
from google_images_download import google_images_download


# Directory Paths
MODEL__DIR = os.path.join(os.getcwd(), "src/server/model_marzocco_detector.h5")
DOWNLD_DIR = os.path.join(os.getcwd(), "src/server/server_images/download")
MODEL__DIR = os.path.join(os.getcwd(), "src/server/model_marzocco_detector.h5")

# Run Time Configurations
IMG_SIZE = 100
HIT_VAL = .01
NUMBER_OF_IMAGE_DOWNLOADS = 20


class DownloadQueue(Thread):

    def __init__(self, imagePredictor):
        Thread.__init__(self)
        self.dwlQueue = queue.Queue()
        self.imgPred = imagePredictor

    def run(self):
        self.__listenToQueue()

    def addToQueue(self, place_id, place_name, place_suffix):
        print("Adding Item To Queue", place_name)
        item = {"place_id": place_id, "place_name": place_name,
                "place_suffix": place_suffix}
        self.dwlQueue.put(item)

    def __listenToQueue(self):
        # SetUp the Model
        model = keras.models.load_model(MODEL__DIR)
        model._make_predict_function()

        err_count = 0

        while True:
            time.sleep(1)
            item = self.dwlQueue.get(block=True)
            dir_path = os.path.join(DOWNLD_DIR, item["place_id"])
            print("Popping Item From Queue")

            try:
                self.__downloadImages(item, dir_path)
                preds = self.imgPred.predictImages(dir_path, model)
                hits = self.__sortImages(item['place_id'], preds)
                #self.saveHitImages(hits, dir_path, save_path)
                self.__deleteImages(dir_path)
                self.__sendPredictions(hits)

            except Exception as e:
                raise e

    def __downloadImages(self, item, dir_path):
        arguments = {
            "output_directory": DOWNLD_DIR,
            "image_directory": item["place_id"],

            "keywords": item["place_name"],
            "suffix_keywords": item["place_suffix"],

            "limit": NUMBER_OF_IMAGE_DOWNLOADS,
            "format": "jpg"
        }

        try:
            response = google_images_download.googleimagesdownload()
            paths = response.download(arguments)
        except Exception as e:
            raise Exception("Error during photo collection")

    def __sortImages(self, place_id, preds):
        hits = []
        for pred, photo_referenece in preds:
            if pred > HIT_VAL:
                hits.append({"place_id": place_id,
                             "marzocco_likelihood": pred, "photo_reference": photo_referenece})
        return hits

    def __saveHitImages(self, hits, dir_path, save_path):
        for pred, photo_referenece in hits:
            if pred > HIT_VAL:
                try:
                    img_path = os.path.join(dir_path, photo_referenece)
                    new_path = os.path.join(save_path, photo_referenece)
                    os.rename(new_path, new_path)
                except Exception as e:
                    raise Exception("Error saving image hits")

    def __deleteImages(self, dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            raise Exception("Error deleting Image")

    def __sendPredictions(self, hits):
        body = {"message": "success", "predictions": hits}
        res = requests.post('http://localhost:3000/predictions', json=body)
        if res.status_code != 200:
            try:
                raise Exception(res.json()['error'])
            except Exception as e:
                raise Exception("Unknown Error sending predictions")
