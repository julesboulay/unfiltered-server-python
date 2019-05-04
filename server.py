from flask import Flask, request, jsonify, Response
import json
import os
import base64
from tensorflow import keras
import numpy as np
import zipfile
import sys

#sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2

PHOTODIR = os.path.join(os.getcwd(), "server_image/photo.jpg")
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
def hello_world():
    if request.method == 'POST':
        # 1. Get JSON base64 data
        content = request.get_json(silent=True)
        buffer = content['data']
        
        # 2. Decode Image and Save
        image_64_decode = base64.b64decode(buffer)
        image_result = open(PHOTODIR, "wb") 
        image_result.write(image_64_decode)

        # 3. Load Image Array
        img_array = cv2.imread(PHOTODIR, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        x = []
        try:
                img_array = cv2.imread(PHOTODIR, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                x.append([new_array])
        except Exception as e:
                pass
        
        # 4. Magic
        x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        pred = model.predict(test_datagen.flow(x, batch_size=1)[0])[0][0].item()

        json_res = {"marzocco_probability": pred}
        return Response(json.dumps(json_res), mimetype='application/json')

if __name__ == '__main__':
    app.run()

