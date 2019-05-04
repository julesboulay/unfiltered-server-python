from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        data = request.get_json(silent=True)
        buffer = data.get("data")
        print(buffer)
        print(len(buffer))
        return jsonify(marzocco_probability=0.8)

if __name__ == '__main__':
    app.run()