from flask.templating import render_template
from keras.backend import print_tensor
from keras.utils.generic_utils import to_list
import tensorflow as tf
import glob
import random
import numpy as np
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint 
from sklearn import metrics  
import time
import keras
import matplotlib.pyplot as plt
import itertools
from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers.core import Dropout,Flatten,Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import confusion_matrix,classification_report
from keras.applications.densenet import DenseNet201
from keras.applications.resnet import ResNet152
from keras.preprocessing.image import ImageDataGenerator
import datetime
import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestRegressor
import json
from flask import Flask, send_from_directory
from tensorflow.python.framework.dtypes import _TYPE_TO_STRING

from werkzeug.utils import redirect

app = Flask(__name__)

data_dir = "./trainTestData/New Plant Diseases Dataset/New Plant Diseases Dataset"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = "./trainTestData/test/test0"
train_datagen_aug = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 20,
                                   horizontal_flip = True)

test_datagen_aug = ImageDataGenerator( rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                rotation_range = 20,
                                horizontal_flip = True)

training_set_aug = train_datagen_aug.flow_from_directory(directory= train_dir,
                                            target_size=(224, 224), 
                                            batch_size=50,
                                            class_mode='categorical')  
label_map = (training_set_aug.class_indices)
#print("Target Classes Mapping Dict:\n")
#print(label_map)

test_set_aug = test_datagen_aug.flow_from_directory(directory= valid_dir,
                                            target_size=(224, 224), # As we choose 64*64 for our convolution model
                                            batch_size=50,
                                            class_mode='categorical',
                                            shuffle=False)
test_set_aug1 = test_datagen_aug.flow_from_directory(directory= test_dir,
                                            target_size=(224, 224), # As we choose 64*64 for our convolution model
                                            batch_size=50,
                                            class_mode='categorical',
                                            shuffle=False)
model = keras.models.load_model('trainedModel.h5')
import os
diseases = os.listdir(train_dir)

app=Flask(__name__,template_folder='template')



@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/aug')
def aug():
    

    return('Augmentation Successfull!')

@app.route('/getDiseases',methods=['GET', 'POST'])
def getDiseases():
    diseaseList = ''
    for disease in diseases:
        diseaseList = diseaseList+disease+":"
    return diseaseList

@app.route('/train')
def prepareModel():
    #for Tensorboard Specific
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    base_model = DenseNet201(include_top=False,
                         input_shape=(224,224,3),
                         weights='imagenet',
                         pooling="avg"
                     )

    base_model.trainable = False
    image_input = Input(shape=(224, 224, 3))

    x = base_model(image_input,training = False)

    x = Dense(256,activation = "relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(128,activation = "relu")(x)
    x = Dropout(0.2)(x)

    image_output = Dense(38,activation="softmax")(x)

    model = Model(image_input,image_output)

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=[tf.keras.metrics.AUC(from_logits=True)])
    model.fit(training_set_aug, epochs=10, validation_data=test_set_aug, callbacks=[callback])
    model.save('trainedModel.h5')
    return('Model Training complete.')

@app.route('/loadModel')
def loadModel():
    model = keras.models.load_model('trainedModel.h5')
    return("Model Loaded successfully")

@app.route('/getChart')
def getChart():
    return('----------------')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    test_set_aug1 = test_datagen_aug.flow_from_directory(directory= test_dir,
                                            target_size=(224, 224), # As we choose 64*64 for our convolution model
                                            batch_size=50,
                                            class_mode='categorical',
                                            shuffle=False)
    x, y = next(test_set_aug1)

    image = x[0]
    
    image = image.reshape(1, 224, 224, 3)
    pred_vec = model.predict(image)        
    pred_max = np.argmax(pred_vec,axis=1) 
    return("The predicted disease is : "+ diseases[pred_max[0]])

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["img"]
            # files = glob.glob('./trainTestData/test/test0/test')
            # for f in files:
            #     os.remove(f)
            image.save(os.path.join("./trainTestData/test/test0/test", image.filename))
            return redirect("predict")
    return ("Error has Occured!!!!")