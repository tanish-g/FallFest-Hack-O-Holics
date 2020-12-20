from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from PIL import Image
import albumentations as aug
from efficientnet_pytorch import EfficientNet

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = torch.load("canc.pth")
model.eval()
def model_predict(file, model):
    image = Image.open(file)
    image = np.array(image)
    transforms = aug.Compose([
            aug.Resize(224,224),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
    image = transforms(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor([image], dtype=torch.float)
    preds = model(image)
    preds = np.argmax(preds.detach())
    return preds

@app.route('/')
def home():
    # Main page
    return render_template('home.html')


@app.route('/imageUpload',methods=['GET'])
def index():
    # image upload page
    return render_template('index.html')

@app.route('/profile',methods=['GET'])
def profile():
    # image upload page
    return render_template('profile.html')

@app.route('/predict', methods=['POST'])
def upload1():
    # Get the file from post request
    f = request.files['file']
    labs=['MELANOMA (MALIGNANT)', 'MELANOCYTIC NEVUS (BENIGN)/ NORMAL SKIN /RASH', 'BASAL CELL CARCINOMA (BENIGN)', 'ACTINIC KERATOSIS (BENIGN)', 'BENIGN KERATOSIS (BENIGN)', 'DERMATOFIBROMA (NON CANCEROUS-BENIGN)', 'VASCULAR LESION (MAYBE BENIGN MAYBE MALIGNANT)', 'SQUAMOUS CELL CARCINOMA(MALIGNANT)']
    # Make prediction
    preds = model_predict(f, model)
    result = labs[preds]
    return result

@app.route('/predict1', methods=['POST'])
def upload():
    # Get the file from post request
    if request.method =='POST':
        skin_lesion=request.get_json()
        image_url=skin_lesion['url']
        urllib.request.urlretrieve(image_url, "sample.png")
        labs=['MELANOMA (MALIGNANT)', 'MELANOCYTIC NEVUS (BENIGN)/ NORMAL SKIN /RASH', 'BASAL CELL CARCINOMA (BENIGN)', 'ACTINIC KERATOSIS (BENIGN)', 'BENIGN KERATOSIS (BENIGN)', 'DERMATOFIBROMA (NON CANCEROUS-BENIGN)', 'VASCULAR LESION (MAYBE BENIGN MAYBE MALIGNANT)', 'SQUAMOUS CELL CARCINOMA(MALIGNANT)']
        preds = model_predict("sample.png", model)
        result = labs[preds]         
    # Make prediction   
    return jsonify({'result':result})


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
