import os
import sys
import shutil
import base64
import json

import numpy as np
import zipfile
import random

from google_images_download import google_images_download
from flask import Flask, request, jsonify, Response
from tensorflow import keras

try:
        sys.path.append('/usr/local/lib/python3.7/site-packages')
except Exception as e:
        pass
import cv2

MODEL__DIR = os.path.join(os.getcwd(), "model_marzocco_detector.h5")
SEARCH_DIR = os.path.join(os.getcwd(), "server_images/search")
MARHIT_DIR = os.path.join(os.getcwd(), "server_images/marzocco_hits")
DOWNLD_DIR = os.path.join(os.getcwd(), "server_images/download")

IMG_SIZE = 100
HIT_VAL = .5

# SetUp the Model
zip_ref = zipfile.ZipFile("model_marzocco_detector.h5.zip", 'r')
zip_ref.extractall(os.getcwd())
zip_ref.close()

model = keras.models.load_model(MODEL__DIR)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# SetUp the Server
app = Flask(__name__)

@app.route('/predictimage', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # 1. Get JSON base64 data
        buffer = ""
        photoid = ""
        if request.data:
                content = request.get_json(silent=True)
                photoid = content['photo_id']
                buffer = content['data']
        else:
                json_res = {"message": "failure", "error": "Missing JSON data"}
                return Response(json.dumps(json_res), mimetype='application/json')
        

        # 2. Decode Image and Save
        photo_path = os.path.join(SEARCH_DIR, photoid + ".jpg")
        try:
                image_64_decode = base64.b64decode(buffer)
                image_result = open(photo_path, "wb")
                image_result.write(image_64_decode)
        except Exception as e:
                json_res = {"message": "failure", "error": "Buffer sent not image compatible"}
                return Response(json.dumps(json_res), mimetype='application/json')

        # 3. Load Image Array
        x = []
        try:
                img_array = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                x.append(new_array)
        except Exception as e:
                json_res = {"message": "failure", "error": "Error loading image into array"}
                return Response(json.dumps(json_res), mimetype='application/json')

        # 4. Magic
        pred = 0
        try:
                x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                pred = model.predict(test_datagen.flow(x, batch_size=1)[0])[0][0].item()
        except Exception as e:
                json_res = {"message": "failure", "error": "Error calculating prediction"}
                return Response(json.dumps(json_res), mimetype='application/json')

        # 5. Save Image If high Prob, else Delete
        '''
        if pred > HIT_VAL:
                try:    
                        new_path = os.path.join(MARHIT_DIR, photoid)     
                        os.rename(photo_path, new_path)
                except Exception as e:
                        pass
        else 
        '''
        try:
                os.remove(photo_path)
        except Exception as e:
                pass
        

        json_res = {"message": "success", "marzocco_probability": pred}
        return Response(json.dumps(json_res), mimetype='application/json')



@app.route('/predictdownload', methods=['POST'])
def predict_download():
    if request.method == 'POST':
        # 1. Get JSON Data from Request
        place_id = ""
        place_name = ""
        place_suffix = ""
        try:
                content = request.get_json(silent=True)
                place_id = content['place_id']
                place_name = content['place_name']
                place_suffix = content['place_suffix']
        except Exception as e:
                json_res = {"message": "failure", "error": "Missing JSON data"}
                return Response(json.dumps(json_res), mimetype='application/json')

        # 2. Download Images from Google
        PLACE_DIR = os.path.join(DOWNLD_DIR, place_id)
        arguments = {
                "output_directory": DOWNLD_DIR,
                "image_directory": place_id,
        
                "keywords": place_name,
                "suffix_keywords": place_suffix,
        
                "limit": 5,
                "format": "jpg" }
        try:
                response = google_images_download.googleimagesdownload()
                paths = response.download(arguments)
        except Exception as e:
                json_res = {"message": "failure", "error": "Error during photo collection"}
                return Response(json.dumps(json_res), mimetype='application/json')

        # 2. Load Image Array
        place_photos = []
        for img in os.listdir(PLACE_DIR):
                try:
                        photo_path = os.path.join(PLACE_DIR, img)
                        img_array = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        place_photos.append([new_array, img])
                except Exception as e:
                        pass

        # 3. Magic (Creating array of predictions)
        preds = []
        for photo, img in place_photos:
                try:
                        x = np.array(photo).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                        pred = model.predict(test_datagen.flow(x, batch_size=1)[0])[0][0].item()
                        preds.append([pred, img])
                except Exception as e:
                        pass

        # 4. Sort through predictions, Move HIT photos to folder for training
        hits = []
        for pred, img in preds:
                if pred > HIT_VAL:
                        try:    
                                hits.append({"marzocco_probability": pred, "img_id": img})
                                '''
                                photo_path = os.path.join(PLACE_DIR, img)
                                new_path = os.path.join(MARHIT_DIR, img)     
                                os.rename(photo_path, new_path)
                                '''
                        except Exception as e:
                                pass

        # 5. Delete PLACE_DIR with rest of photos
        for img in os.listdir(PLACE_DIR):
                try:
                        shutil.rmtree(PLACE_DIR)
                except Exception as e:
                        pass

        # 6. Return Hits
        if len(place_photos) < 1:
                json_res = {"message": "failure", "error": "No photo succesfully downloaded and loaded"}
                return Response(json.dumps(json_res), mimetype='application/json')
        elif len(preds) < 1:
                json_res = {"message": "failure", "error": "No photo succesfully processed by model"}
                return Response(json.dumps(json_res), mimetype='application/json')
        else:
                json_res = {"message": "success", "predictions": hits}
                return Response(json.dumps(json_res), mimetype='application/json')



@app.route('/predictmock', methods=['POST'])
def predict_mock():
    if request.method == 'POST':
        mock_pred = random.random()
        json_res = {"message": "success", "marzocco_probability": mock_pred}
        return Response(json.dumps(json_res), mimetype='application/json')


if __name__ == '__main__':
    app.run()
