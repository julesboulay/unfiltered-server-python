from flask import Flask, request, jsonify, Response
import json

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        content = request.get_json(silent=True)
        buffer = content['data']
        print("Lenght of Buffer: " + str(len(buffer)))
        
        json_res = {"marzocco_probability": 0.8}
        return Response(json.dumps(json_res), mimetype='application/json')

if __name__ == '__main__':
    app.run()