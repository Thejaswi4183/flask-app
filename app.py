from flask import Flask, request, jsonify

app = Flask(__name__)

# Root route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running!"})

# POST route
@app.route("/echo", methods=["POST"])
def echo():
    data = request.get_json()
    return jsonify({"you_sent": data})

if __name__ == "__main__":
    app.run(debug=True)
