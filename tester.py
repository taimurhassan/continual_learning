import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import ResNet101
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.applications.resnet50 import *
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import optimizers
import keras.backend as K

numClasses = 5
class_names = ['normal','DME','CNV','DRUSEN','CSR']
    
model=load_model("model2.h5",custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})

folder = "./datasets/test/"

for filename in os.listdir(folder):
    image = load_img(os.path.join(folder,filename), target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    print(yhat)
    index = np.argmax(yhat)
    print(class_names[index])
