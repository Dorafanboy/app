# -*- coding: utf-8 -*-
from __future__ import division, print_function
from wand.image import Image
from PIL import Image
from flask import Flask, request, \
        render_template, redirect, url_for,\
        session, send_file

# Flask WTF FORMS
from flask_wtf import FlaskForm,RecaptchaField
from wtforms import (StringField,SubmitField,
                     DateTimeField, RadioField,
                     SelectField,TextAreaField, DateField)
import lxml.etree as ET
from wtforms.validators import DataRequired
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import glob
import re
import cv2
# import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#keras
import tensorflow.keras as krs
# from keras import Model
# from keras.layers import Input
# from tensorflow.keras.models import load_model
# from keras.applications.resnet50 import preprocess_input, decode_predictions


#xml
from xml.etree import ElementTree


app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"

app.config["RECAPTCHA_PUBLIC_KEY"] = "6LcN1ucaAAAAACyMzmU6Tzy_DshySBfzdpxQdTHJ"
app.config["RECAPTCHA_PRIVATE_KEY"] = "6LcN1ucaAAAAAJSFWsyQjD4wq4REfhrvhftIELuw"


class Widgets(FlaskForm):
    recaptcha = RecaptchaField()

    name = StringField(label="Name", validators=[DataRequired()])

    radio = RadioField(label ="Please select Your Programming language ",
                       choices=[('Python', "Python"), ["C++","C++"]])

    submit = SubmitField(label="Submit")


@app.route("/", methods=("GET", "POST"))
def home():
    form = Widgets()
    if request.method == "POST":
        if form.validate_on_submit():
            session["name"] = form.name.data
            print("Name Entered {}".format(form.name.data))
            return redirect(url_for('result'))

    if request.method == "GET":
        return render_template("list.html", form=form)


@app.route("/result", methods=["GET", "POST"])
def result():
    return "Thanks {}".format(session["name"])
# MAIN NEIRON
model = ResNet50(weights='imagenet')
# model.save('DogCat.h5')
MODEL_PATH = 'DogCat.h5'

# Load your trained model
# model.load_weights(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

print('Model loaded. Check http://127.0.0.1:5000/')
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('list.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result

@app.route("/",methods=['GET','POST'])
def apixml():
    #парсим xml файл в dom
    dom = ET.parse("file.xslt")
    #парсим шаблон в dom
    xslt = ET.parse("file.xml")
    #получаем трансформер
    transform = ET.XSLT(xslt)
    #преобразуем xml с помощью трансформера xslt
    newhtml = transform(dom)
    #преобразуем из памяти dom в строку, возможно, понадобится указать кодировку
    strfile = ET.tostring(newhtml)
    return strfile
@app.route('/img', methods=['GET', 'POST'])
def intensivity():
    if request.method == 'POST':
        img = mpimg.imread('C:\\Users\\днс\\Desktop\\qrwfsfsfs\\pythonnn\\uploads\\1.jpg')
        lum_img = img[:, :, 0]
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(img)
        ax.set_title('Before')
        plt.colorbar(ticks=[1, 50, 150, 250], orientation='horizontal')
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(lum_img)
        imgplot.set_clim(100.0, 0.7)
        ax.set_title('After')
        x = plt.colorbar(ticks=[1, 50, 100, 200], orientation='horizontal')
        return x
if __name__ == "__main__":
    app.run(debug=True)


