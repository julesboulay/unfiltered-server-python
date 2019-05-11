import os
import json
from flask import Flask, request, jsonify, Response
from src.server.downloadQueue import DownloadQueue
from src.server.imagePredictor import ImagePredictor


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
            pred = imgPred.predictImage(photo_reference, buffer)
        except Exception as e:
            json_res = {"message": "failure", "error": "Missing JSON data"}
            return Response(json.dumps(json_res), mimetype='application/json')

        # Send Response Probability
        json_res = {"message": "success", "marzocco_probability": pred}
        return Response(json.dumps(json_res), mimetype='application/json')


@app.route('/predictdownload', methods=['POST'])
def predict_download():
    if request.method == 'POST':
        # Get JSON Data from Request
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

        # Check If Data Valid
        if not place_id or not place_name or not place_suffix:
            json_res = {"message": "failure", "error": "Missing JSON data"}
            return Response(json.dumps({"message": "success"}), mimetype='application/json')
        else:
            # Add To Queue And Send Response
            dwlQueue.addToQueue(place_id, place_name, place_suffix)
            json_res = {"message": "success"}
            return Response(json.dumps(json_res), mimetype='application/json')


@app.route('/predictmock', methods=['POST'])
def predict_mock():
    if request.method == 'POST':
        mock_pred = random.random()
        json_res = {"message": "success", "marzocco_probability": mock_pred}
        return Response(json.dumps(json_res), mimetype='application/json')


if __name__ == '__main__':
    app.run()
