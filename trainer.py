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

doTraining = True
numClasses = 5
class_names = ['normal','DME','CNV','DRUSEN','CSR']
numIterations = numClasses
adaptationIteration = 0

def mutualDistillationLoss(yTrue, yPred, oldClasses, newClasses):
    yOldP = yPred[:,:oldClasses]
    yOldT = yTrue[:,:oldClasses]
    yNewP = yPred[:,newClasses:]
    yNewT = yTrue[:,newClasses:]
        
    print(yNewP)
    print(yOldP)
    yop = yOldP
    yot = yOldT
    ynp = yNewP
    ynt = yNewT
        
    n1 = float(yOldP.get_shape().as_list()[1])
    n2 = float(yNewP.get_shape().as_list()[1])
            
    pOld = n1/(n1+n2)
    pNew = n2/(n1+n2)
            
    yOldP = (yOldP * yNewP) / yNewP
    m1 = K.mean(yOldP)
    s1 = K.std(yOldP)
    like1 = (1./(np.sqrt(2.*3.1415)*s1))*K.exp(-0.5*((yOldP-m1)**2./(s1**2.)))
            
    yOldNewP = like1 * pOld  
    
    return K.mean(K.categorical_crossentropy(yOldT,yOldNewP))
    
def continualLearningLoss(yTrue,yPred, iteration, oldClasses,temperature):
    a = 0.25
    b = 0.45
    c = 0.30
    if iteration == 0:
        return K.categorical_crossentropy(yTrue,yPred,from_logits=True)
    else:
        newClasses = 1
        if iteration == adaptationIteration:
            newClasses = 2
        total = oldClasses
        oldClasses = oldClasses - newClasses    
        
        yOldP = yPred[:,:oldClasses]/ temperature
        yOldT = yTrue[:,:oldClasses]
        yNewP = yPred[:,newClasses:]/ temperature
        yNewT = yTrue[:,newClasses:]
        
        return (a * K.categorical_crossentropy(yOldP,yOldT,from_logits=True)) + (b * mutualDistillationLoss(yTrue, yPred, oldClasses, newClasses)) + (c * keras.losses.kullback_leibler_divergence(yNewT,yNewP,from_logits=True))
        
if doTraining == True:
    c = 0 # classes to add
    for i in range(0,numIterations):
    
        trainingPath = "./datasets/train" + str(i + 1) + "/"
        if i == 0 or i == adaptationIteration:
            c = c + 2 # 2 classes are added in the start, and at the adaptation stage
            base_model=MobileNet(weights='imagenet',include_top=False) 

            x=base_model.output
            x=GlobalAveragePooling2D()(x)
            x=Dense(1024,activation='relu')(x)
            x=Dense(1024,activation='relu')(x)
            x=Dense(512,activation='relu')(x) 
            preds=Dense(c,activation='softmax')(x) 
            model=Model(inputs=base_model.input,outputs=preds)
        else:
            c = c + 1
            print(c)
            model.layers.pop()
            preds=Dense(c,activation='softmax')(model.layers[-1].output) 
            model=Model(inputs=model.input,outputs=preds)

        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

        train_generator=train_datagen.flow_from_directory(trainingPath, 
                                                         target_size=(224,224),
                                                         color_mode='rgb',
                                                         batch_size=8,
                                                         class_mode='categorical',
                                                         shuffle=True)

        temperature = 1.65
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer='adadelta',loss=lambda yTrue, yPred: continualLearningLoss(yTrue, yPred, i, c, temperature),metrics=['accuracy'])

        step_size_train=train_generator.n//train_generator.batch_size
        fit_history = model.fit_generator(generator=train_generator,
                           steps_per_epoch=step_size_train, epochs=20)

        
        model.save("model" + str(i + 1) + ".h5")


#model_name = "model2.h5"    
#model=load_model(model_name,custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
#image = load_img('test.jpg', target_size=(224, 224))
#image = img_to_array(image)
#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#image = preprocess_input(image)
#yhat = model.predict(image)
#print(yhat)
#index = np.argmax(yhat)
#print(class_names[index])
