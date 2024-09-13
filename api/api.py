from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search_image():
    return jsonify({"message": "API Python dang chay tren Vercel!"})
@app.route('/')
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)
