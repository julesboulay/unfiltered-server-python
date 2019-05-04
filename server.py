from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        buffer = request.json.get("data")
        print("Lenght of Buffer: " + str(len(buffer)))
        return jsonify(marzocco_probability=0.8)

if __name__ == '__main__':
    app.run()