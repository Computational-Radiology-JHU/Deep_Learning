# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:46:28 2021

@author: vishwa
"""

from __future__ import print_function, division
import scipy

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, MaxPool3D,MaxPool2D
from tensorflow.keras.layers import LeakyReLU, Lambda, Multiply, ReLU, GlobalAveragePooling3D, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow.keras.layers.experimental.preprocessing import RandomContrast
import tensorflow.keras.losses
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import tensorflow.keras.metrics as mtrc
import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.io import imsave
import scipy.io
import warnings

class Classifier_3D:
    def __init__(self,model_type = 'vanilla', imSize=[512,512,64,1], labelSize=None):
        
        self.img_shape = imSize
        self.label_shape = labelSize
        self.gf = 32
        self.num_classes = labelSize[-1]
        #self.num_channels = imSize[-1]
        
        classifyNet,lossFn,metricsFn = self.getClassificationModel(model_type)        
        self.model = classifyNet()
        print(self.model)
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss=lossFn,
            optimizer=optimizer,
             metrics=metricsFn)
                
        
    
    def getClassificationModel(self,unet_type):
        allClassifiers = {'vanilla3D':self.classifier3D,
                    'vanilla2D':self.classifier2D,
                    
                    # 'resnet':self.predefinedModel('resnet'),
                    # 'inceptionv4':self.predefinedModel('inceptionv4'),
                    # 'densenet':self.predefinedModel('denseNet'),                    
                    }
        # allLoss = {'vanilla':"categorical_crossentropy",
        #             'sparse':"categorical_crossentropy",
        #             'unsupervised':"mean_squared_error",
        #             'WNet':['categorical_crossentropy','mean_squared_error'],
        #             #'unsupervised':self.unsupervisedUNet,
        #             #'unet++':self.unet_plus_plus,
        #             }
        # allMetrics = {'vanilla':"accuracy",
        #             'sparse':"accuracy",
        #             'unsupervised':"accuracy",
        #             'WNet':['accuracy','mean_squared_error'],
        #             #'unsupervised':self.unsupervisedUNet,
        #             #'unet++':self.unet_plus_plus,
        #             } 
        if self.num_classes<2:
            return allClassifiers[unet_type],'binary_crossentropy',["accuracy"]
        else:
            return allClassifiers[unet_type],'categorical_crossentropy',["accuracy"]
    
    def classifier3D(self):
        """3D classifier for volumetric input data """

        
        # Image input
        inputs = Input(shape=self.img_shape)
        

        x = Conv3D(filters=16, kernel_size=3, activation="relu")(inputs)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
    
        x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
    
        x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
    
#        x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#        x = MaxPool3D(pool_size=2)(x)
#        x = BatchNormalization()(x)
    
        x = GlobalAveragePooling3D()(x)
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.3)(x)
        if self.num_classes == 1:
            outputs = Dense(units=self.num_classes, activation="sigmoid")(x)
        else:
            outputs = Dense(units=self.num_classes, activation="softmax")(x)

        return Model(inputs, outputs)
    
    def classifier2D(self):
        """2D classifier for image input data """

        
        # Image input
        inputs = Input(shape=self.img_shape)
        #y = RandomContrast((0.25, 0.25))(inputs)

        x = Conv2D(filters=16, kernel_size=3, activation="relu")(inputs)
        x = MaxPool2D(pool_size=2)(x)
        x = BatchNormalization()(x)
    
        x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = MaxPool2D(pool_size=2)(x)
        x = BatchNormalization()(x)
    
        x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = MaxPool2D(pool_size=2)(x)
        x = BatchNormalization()(x)
    
#        x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#        x = MaxPool3D(pool_size=2)(x)
#        x = BatchNormalization()(x)
    
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.3)(x)
        if self.num_classes == 1:
            outputs = Dense(units=self.num_classes, activation="sigmoid")(x)
        else:
            outputs = Dense(units=self.num_classes, activation="softmax")(x)

        return Model(inputs, outputs)
    
    