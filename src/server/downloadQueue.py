import os
import time
import requests
import shutil
import queue
from threading import Thread
from google_images_download import google_images_download

from imagePredictor import ImagePredictor

# Directory Paths
DOWNLD_DIR = os.path.join(os.getcwd(), "src/server/server_images/download")

# Run Time Configurations
IMG_SIZE = 100
HIT_VAL = .5
NUMBER_OF_IMAGE_DOWNLOADS = 4


class DownloadQueue(Thread):

    def __init__(self, imagePredictor):
        Thread.__init__(self)
        self.dwlQueue = queue.Queue()
        self.imgPred = imagePredictor

    def run(self):
        self.__listenToQueue()

    def addToQueue(self, place_id, place_name, place_suffix):
        item = {"place_id": place_id, "place_name": place_name,
                "place_suffix": place_suffix}
        self.dwlQueue.put(item)

    def __listenToQueue(self):
        err_count = 0
        while True:
            if self.dwlQueue.empty():
                # time.sleep(.50)
                time.sleep(2)
                print("listening")
            else:
                item = self.dwlQueue.get()
                dir_path = os.path.join(DOWNLD_DIR, item["place_id"])

                try:
                    self.__downloadImages(item, dir_path)
                    preds = self.imgPred.predictImages(dir_path)
                    hits = self.__sortImages(item, preds)
                    #saveHitImages(hits, dir_path, save_path)
                    self.__deleteImages(dir_path)
                    # self.__sendPredictions(hits)
                    print(preds)
                except Exception as e:
                    raise e

    def __downloadImages(self, item, dir_path):
        arguments = {
            "output_directory": DOWNLD_DIR,
            "image_directory": item["place_id"],

            "keywords": item["place_id"],
            "suffix_keywords": item["place_suffix"],

            "limit": NUMBER_OF_IMAGE_DOWNLOADS,
            "format": "jpg"
        }

        try:
            response = google_images_download.googleimagesdownload()
            paths = response.download(arguments)
        except Exception as e:
            raise Exception("Error during photo collection")

    def __sortImages(self, item, preds):
        hits = []
        for pred, photo_referenece in preds:
            if pred > HIT_VAL:
                hits.extend({"place_id": item["place_id"],
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
            raise Exception("Error sending predictions")


imgPred = ImagePredictor()
dwlQueue = DownloadQueue(imgPred)
dwlQueue.start()

time.sleep(10)
dwlQueue.addToQueue("place_id_1", "Winston's Coffee", "Kennedy Town")
