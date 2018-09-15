#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:47:41 2018

@author: mshokry
"""
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

app = flask.Flask(__name__)
model = None

def load_model():
    global model
    model = inception_v3.InceptionV3(weights='imagenet')

def prepare_image(image, size):
    # EGB, Resize, Array, Expand Dim
    if image.mode != 'RGB':
       image = image.convert('RGB')
    image = image.resize(size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = inception_v3.preprocess_input(image)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img))
            img = prepare_image(img,size=(299,299))
            pred = model.predict(img)
            print (pred)
    return flask.jsonify(data)

print(("* Loading Keras model and Flask starting server..."
	"please wait until server has fully started"))
load_model()
app.run()
