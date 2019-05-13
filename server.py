import os
import json
import zipfile
import random
from flask import Flask, request, jsonify, Response
from src.downloadqueue import DownloadQueue
from src.imagepredictor import ImagePredictor

# Directory Paths
SEARCH_DIR = os.path.join(os.getcwd(), "src/server/server_images/search")

# SetUp the Model
zip_ref = zipfile.ZipFile("src/server/model_marzocco_detector.h5.zip", 'r')
zip_ref.extractall(os.getcwd())
zip_ref.close()

# Start the Queue Thread
imgPred = ImagePredictor()
dwlQueue = DownloadQueue(imgPred)
dwlQueue.start()

# SetUp the Server
app = Flask(__name__)


@app.route('/predictimage', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Get JSON Data from Request
        buffer = ""
        photo_reference = ""
        try:
            content = request.get_json(silent=True)
            photo_reference = content['photo_reference']
            buffer = content['data']
        except Exception as e:
            json_res = {"message": "failure", "error": "Missing JSON data"}
            return Response(json.dumps(json_res), mimetype='application/json')

        # Calculate Marzocco Probability
        pred = 0
        try:
            img_path = os.path.join(SEARCH_DIR, photo_reference + ".jpg")
            imgPred.saveImage(img_path, buffer)
            pred = imgPred.predictImage(img_path)
            imgPred.deleteImage(img_path)
        except Exception as e:
            json_res = {"message": "failure", "error": str(e)}
            return Response(json.dumps(json_res), mimetype='application/json')

        # Send Response Probability
        json_res = {"message": "success", "marzocco_probability": pred}
        return Response(json.dumps(json_res), mimetype='application/json')


@app.route('/predictdownload', methods=['POST'])
def predict_download():
    if request.method == 'POST':

        # Get JSON Data from Request
        places = []
        try:
            content = request.get_json(silent=True)
            places = content['places']
        except Exception as e:
            json_res = {"message": "failure", "error": "Missing JSON data"}
            return Response(json.dumps(json_res), mimetype='application/json')

        # Add To Queue And Send Response
        try:
            for p in places:
                dwlQueue.addToQueue(
                    p['place_id'], p['place_name'], p['place_suffix'])
        except Exception as e:
            json_res = {"message": "failure", "error": "Missing JSON data"}
            return Response(json.dumps(json_res), mimetype='application/json')

        json_res = {"message": "success"}
        return Response(json.dumps(json_res), mimetype='application/json')


@app.route('/predictmock', methods=['POST'])
def predict_mock():
    if request.method == 'POST':
        mock_pred = random.random()
        json_res = {"message": "success", "marzocco_probability": mock_pred}
        return Response(json.dumps(json_res), mimetype='application/json')


if __name__ == '__main__':
    app.run(threaded=True)
