from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Path where the celebrity images are stored
CELEBRITY_IMAGE_FOLDER = os.path.join(os.getcwd(), 'thaicelebrities')

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Get the image from the request
    file = request.files['image']
    filename = secure_filename(file.filename)

    # Save the uploaded image to a temporary directory
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Preprocess the uploaded image
    uploaded_image = preprocess_image(file_path)
    
    # Extract features from the uploaded image
    uploaded_features = model.predict(uploaded_image)

    best_match = None
    best_similarity = -1

    # Loop through celebrity images in the folder and calculate similarity
    for celebrity_image_name in os.listdir(CELEBRITY_IMAGE_FOLDER):
        celebrity_image_path = os.path.join(CELEBRITY_IMAGE_FOLDER, celebrity_image_name)

        # Preprocess each celebrity image
        celebrity_image = preprocess_image(celebrity_image_path)
        
        # Extract features from the celebrity image
        celebrity_features = model.predict(celebrity_image)

        # Calculate similarity (Cosine Similarity)
        similarity = cosine_similarity(uploaded_features, celebrity_features)[0][0]

        # Track the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = celebrity_image_name

    # Prepare the response
    result = {
        "predicted_class": best_match,
        "confidence_score": round(float(best_similarity) * 100, 2)
    }

    # Remove the saved uploaded image after prediction
    os.remove(file_path)

    return jsonify(result), 200

# Start the Flask app
if __name__ == '__main__':
    # Create the 'uploads' folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
