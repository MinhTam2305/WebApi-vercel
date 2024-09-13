import os
import numpy as np
import pickle
import requests
from flask import Flask, request, jsonify
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Firebase Storage URL for the features.pkl file
FIREBASE_FILE_URL = "https://firebasestorage.googleapis.com/v0/b/adoptpet-flutter.appspot.com/o/features.pkl?alt=media&token=cbf0aacf-e8bb-4449-a398-4656a5a62bcc"

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to download the file from Firebase
def download_file_from_firebase(download_path):
    response = requests.get(FIREBASE_FILE_URL)
    if response.status_code == 200:
        with open(download_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    return features.flatten()

# Function to load saved features
def load_saved_features():
    features_path = '/tmp/features.pkl'
    # Check if the file doesn't exist, download it
    if not os.path.exists(features_path):
        download_file_from_firebase(features_path)
    
    # Load the features from the file
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

# Function to find the most similar image
def find_most_similar_image(query_image_path, features_dict):
    query_features = extract_features(query_image_path)
    
    max_similarity = -1
    most_similar_image_path = None
    
    for img_path, data_features in features_dict.items():
        similarity = cosine_similarity([query_features], [data_features])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image_path = img_path
    
    return most_similar_image_path, max_similarity

# API route to find similar images
@app.route('/find_similar', methods=['POST'])
def find_similar_image():
    data = request.json
    query_image_path = data.get('image_path')
    
    if not query_image_path:
        return jsonify({"error": "No image path provided"}), 400
    
    features_dict = load_saved_features()
    
    # Save the uploaded image to a temporary location
    temp_image_path = '/tmp/query_image.png'
    with open(temp_image_path, 'wb') as f:
        f.write(request.files['image'].read())
    
    if not os.path.exists(temp_image_path):
        return jsonify({"error": "Query image not found"}), 404
    
    most_similar_image_path, similarity = find_most_similar_image(temp_image_path, features_dict)
    
    if most_similar_image_path:
        return jsonify({
            "most_similar_image": most_similar_image_path,
            "similarity": similarity
        }), 200
    else:
        return jsonify({"error": "No similar image found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
