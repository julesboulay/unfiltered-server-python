from flask import Flask, request, jsonify, Response

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        buffer = request.json.get("data")
        print("Lenght of Buffer: " + str(len(buffer)))
        
        json_res = jsonify(marzocco_probability=0.8)
        return Response(json_res, mimetype='application/json')

if __name__ == '__main__':
    app.run()