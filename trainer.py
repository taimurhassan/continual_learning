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
numClasses = 2
class_names = ['normal','ci-DME']
numIterations = 2

def mutualDistillationLoss(yTrue, yPred, oldClasses):
    yOldP = yPred[:,:oldClasses]
    yOldT = yTrue[:,:oldClasses]
    yNewP = yPred[:,oldClasses-1:]
    yNewT = yTrue[:,oldClasses-1:]
        
    n1 = float(yOldP.get_shape().as_list()[1])
    n2 = float(yNewP.get_shape().as_list()[1])
        
    pOld = n1/(n1+n2)
    pNew = n2/(n1+n2)
        
    yOldP = (yOldP * yNewP) / yNewP
    m1 = K.mean(yOldP)
    s1 = K.std(yOldP)
    like1 = (1./(np.sqrt(2.*3.1415)*s1))*K.exp(-0.5*((yOldP-m1)**2./(s1**2.)))
        
    # evidence is optional
    yOldNewP = like1 * pOld 
    
    yOldP = yPred[:,:oldClasses-1]
    yOldT = yTrue[:,:oldClasses-1]
    yNewP = yPred[:,oldClasses:]
    yNewT = yTrue[:,oldClasses:]
    yNewP = (yOldP * yNewP) / yOldP 
    m2 = K.mean(yNewP)
    s2 = K.std(yNewP)       
    like2 = (1./(np.sqrt(2.*3.1415)*s2))*K.exp(-0.5*((yNewP-m2)**2./(s2**2.)))
    
    yNewP = like2 * pNew 
    
    return K.categorical_crossentropy(yOldT,yOldNewP) + K.categorical_crossentropy(yNewT,yNewP)
    
def incrementalLearningLoss(yTrue,yPred, iteration, oldClasses,temperature):
    if iteration == 0:
        return K.categorical_crossentropy(yTrue,yPred,from_logits=True)
    else:
        yOldP = yPred[:,:oldClasses]/temperature
        yOldT = yTrue[:,:oldClasses]/temperature
        yNewP = yPred[:,oldClasses:]/temperature
        yNewT = yTrue[:,oldClasses:]/temperature
        
        return K.categorical_crossentropy(yNewT,yNewP) + mutualDistillationLoss(yTrue, yPred, oldClasses) + keras.losses.kullback_leibler_divergence(yNewT,yNewP)
        
if doTraining == True:
    
    c = 1 # classes to add
    for i in range(0,numIterations):
    
        trainingPath = "./datasets/train" + str(i + 1) + "/"
        if i is 0:
            base_model=MobileNet(weights='imagenet',include_top=False) 

            x=base_model.output
            x=GlobalAveragePooling2D()(x)
            x=Dense(1024,activation='relu')(x)
            x=Dense(1024,activation='relu')(x)
            x=Dense(512,activation='relu')(x) 
            preds=Dense(c,activation='softmax')(x) 
            model=Model(inputs=base_model.input,outputs=preds)
        else:
            print(c)
            model.layers.pop()
            preds=Dense(c,activation='softmax')(model.layers[-1].output) 
            model=Model(inputs=model.input,outputs=preds)

        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

        train_generator=train_datagen.flow_from_directory(trainingPath, 
                                                         target_size=(224,224),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True)

        temperature = 2.
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer='adadelta',loss=lambda yTrue, yPred: incrementalLearningLoss(yTrue, yPred, i, c, temperature),metrics=['accuracy'])

        step_size_train=train_generator.n//train_generator.batch_size
        fit_history = model.fit_generator(generator=train_generator,
                           steps_per_epoch=step_size_train,
                           epochs=20)
            
        
#        plt.subplot(121)  
#        plt.plot(fit_history.history['acc'])  
        #plt.plot(fit_history.history['val_acc'])  
#        plt.title('model accuracy')  
#        plt.ylabel('accuracy')  
#        plt.xlabel('epoch')  
        #plt.legend(['train', 'valid']) 
            
#        plt.subplot(122)  
#        plt.plot(fit_history.history['loss'])  
        #plt.plot(fit_history.history['val_loss'])  
#        plt.title('model loss')  
#        plt.ylabel('loss')  
#        plt.xlabel('epoch')  
        #plt.legend(['train', 'valid']) 

#        plt.show()

        c = c + 1
        model.save("model" + str(i + 1) + ".h5")


    
model=load_model("model2.h5",custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
image = load_img('test.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
yhat = model.predict(image)
print(yhat)
index = np.argmax(yhat)
print(class_names[index])
