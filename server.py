from flask import Flask, request, jsonify, Response
import json
import os
import base64
from tensorflow import keras
import numpy as np
import zipfile
import sys
import random
#sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2

PHOTODIR = os.path.join(os.getcwd(), "server_image")
MODELDIR = os.path.join(os.getcwd(), "model_marzocco_detector.h5")
IMG_SIZE = 100

# SetUp the Model
zip_ref = zipfile.ZipFile("model_marzocco_detector.h5.zip", 'r')
zip_ref.extractall(os.getcwd())
zip_ref.close()

model = keras.models.load_model(MODELDIR)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# SetUp the Server
app = Flask(__name__)

@app.route('/', methods=['POST'])
def poredict_image():
    if request.method == 'POST':
        # 1. Get JSON base64 data
        buffer = ""
        photoid = ""
        if request.data:
                content = request.get_json(silent=True)
                photoid = content['photo_id']
                buffer = content['data']
        else:
                json_res = {"message": "failure", "error": "No data sent"}
                return Response(json.dumps(json_res), mimetype='application/json')
        

        # 2. Decode Image and Save
        photo_path = os.path.join(PHOTODIR, photoid + ".jpg")
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
                x.append([new_array])
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

        json_res = {"message": "success", "marzocco_probability": pred}
        return Response(json.dumps(json_res), mimetype='application/json')


@app.route('/mock', methods=['POST'])
def poredict_image_mock():
    if request.method == 'POST':
        mock_pred = random.random()
        json_res = {"message": "success", "marzocco_probability": mock_pred}
        return Response(json.dumps(json_res), mimetype='application/json')


if __name__ == '__main__':
    app.run()
