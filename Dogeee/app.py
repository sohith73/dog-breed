from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)
model = load_model('dog_breed.h5')
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.resize(opencv_image, (224,224))
            opencv_image.shape = (1,224,224,3)
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            return result

if __name__ == '__main__':
    app.run(debug=True)
