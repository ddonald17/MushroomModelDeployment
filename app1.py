from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
import imageio
import cv2

# Define a flask app
app = Flask(__name__)

# Model saved with PyTorch torch.save()
MODEL_PATH = 'model.pt'

# Load your trained model
model = torch.load(MODEL_PATH)
model.eval()

# Class labels
class_labels = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

def model_predict(img_path, model):
    #img = cv2.imread('img_path')
    #img = cv2.resize(img, (224, 224))
    img = load_img(img_path, target_size=(224, 224))
    #img = imageio.imread(img_path)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')
    x /= 255
    x = torch.from_numpy(x).permute(0, 3, 1, 2)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    with torch.no_grad():
        preds = model(x)
        preds = torch.softmax(preds, dim=1)
    return preds.numpy()[0]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for genus
        pred_class = np.argmax(preds) 
        result = class_labels[pred_class]
        if(pred_class == 1 or pred_class==0 ):
            edible = "Edible"
        else:
            edible = "Not edible"
        return result,edible
    return None

if __name__ == '__main__':
    app.run(debug=True)
