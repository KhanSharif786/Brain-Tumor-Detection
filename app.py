import os
from flask import Flask, jsonify, request
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import normalize
from flask_cors import CORS
import cv2


app = Flask(__name__)
CORS(app)

model = load_model('cnn_model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "no_tumor"
    elif classNo == 1:
        return "meningioma_tumor"
    elif classNo == 2:
        return "glioma_tumor"
    elif classNo == 3:
        return "pituitary_tumor"
    elif classNo == 4:
        return "gh"
    elif classNo ==5:
        return "skdjh"
    elif classNo ==6:
        return "jkdhjg"
    elif classNo ==17:
        return "jkdhjg"
    elif classNo ==7:
        return "jkdhjg"
    elif classNo ==8:
        return "jkdhjg"
    elif classNo ==9:
        return "jkdhjg"
    elif classNo ==10:
        return "jkdhjg"
    elif classNo ==11:
        return "jkdhjg"
    elif classNo ==12:
        return "jkdhjg"
    elif classNo ==13:
        return "jkdhjg"
    elif classNo ==14:
        return "jkdhjg"
    elif classNo ==16:
        return "jkdhjg"
    elif classNo ==18:
        return "jkdhjg"
    elif classNo ==19:
        return "jkdhjg"
    

@app.route('/', methods=['GET'])
def index():
    return "<p>Hello, World!</p>"

@app.route('/predict', methods=['POST'])
def upload():
     if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        if file:
            # img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            # img = cv2.resize(img, (64, 64))
            img_array = np.array(file)
            # img_array = normalize(img_array, axis=1)  
            # img_array = np.expand_dims(img_array, axis=0)
            img_array = image.img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            result = model.predict(img_array) 
            class_label = np.argmax(result)
            class_name = get_className(class_label)
            return jsonify({"result": class_name, "image":file.filename})
        return "No Prediction"

if __name__ == '__main__':
    app.run(debug=True)