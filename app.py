from flask import Flask, render_template, request, redirect
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the pre-trained emotion recognition model
model = load_model('face_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(request.url)
    
    # Read the image and preprocess it
    img = Image.open(file.stream)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to the size expected by the model
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

    # Predict emotion
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]

    return render_template('result.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)