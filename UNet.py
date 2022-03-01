# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:46:28 2021

@author: vishwa
"""

from __future__ import print_function, division
import scipy

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import LeakyReLU, Lambda, Multiply, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv3D, UpSampling3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
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

class UNet:
    def __init__(self,unet_type = 'vanilla', imSize=[512,512,1], labelSize=[512,512,6]):
        
        self.img_shape = imSize
        self.label_shape = labelSize
        self.gf = 32
        self.num_classes = labelSize[-1]
        self.num_channels = imSize[-1]
        print(self.num_classes)
        unet,lossFn,metricsFn = self.getUNet(unet_type)        
        self.model = unet()
        print(self.model)
        optimizer = Adam(0.0002, 0.5)
        print(metricsFn)
        self.model.compile(loss=lossFn,
            optimizer=optimizer,
             metrics=metricsFn)
                
        
    
    def getUNet(self,unet_type):
        allUNets = {'vanilla':self.vanillaUNet,
                    'vanilla3D':self.vanilla3DUNet,
                    'sparse':self.sparseUNet,
                    'unsupervised':self.unsupervedUNet,
                    'WNet':self.WNet,
                    'S4Net':self.S4Net,
                    #'unsupervised':self.unsupervisedUNet,
                    #'unet++':self.unet_plus_plus,
                    }
        if self.num_classes>=2:
            allLoss = {'vanilla':"categorical_crossentropy",
                       'vanilla3D':"categorical_crossentropy",
                        'sparse':"categorical_crossentropy",
                        'unsupervised':"mean_squared_error",
                        'WNet':['categorical_crossentropy','mean_squared_error'],
                        'S4Net':['mean_absolute_error','mean_absolute_error'],
                        #'unsupervised':self.unsupervisedUNet,
                        #'unet++':self.unet_plus_plus,
                        }
        else:
            allLoss = {'vanilla':"binary_crossentropy",
                       'vanilla3D':"binary_crossentropy",
                        'sparse':"binary_crossentropy",
                        'unsupervised':"mean_squared_error",
                        'WNet':['binary_crossentropy','mean_squared_error'],
                        'S4Net':['mean_absolute_error','mean_absolute_error'],
                        #'unsupervised':self.unsupervisedUNet,
                        #'unet++':self.unet_plus_plus,
                        }
        allMetrics = {'vanilla':["accuracy"],
                    'vanilla3D':["accuracy"],
                    'sparse':["accuracy"],
                    'unsupervised':["accuracy"],
                    'WNet':['accuracy','mean_squared_error'],
                    'S4Net':['mean_absolute_error','mean_absolute_error'],
                    #'unsupervised':self.unsupervisedUNet,
                    #'unet++':self.unet_plus_plus,
                    }        
        print(allLoss[unet_type],allMetrics[unet_type])
        return allUNets[unet_type],allLoss[unet_type],allMetrics[unet_type]
    
    def vanillaUNet(self):
        """U-Net for fully labeled data """

        
        # Image input
        d0 = Input(shape=self.img_shape)
        #dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv2d(d0, self.gf,isPool=False) #Size=512x512
        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv2D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        
        if self.num_classes < 2:
            output_classes = Dense(self.num_classes, activation='sigmoid')(output_img)
        else:
            output_classes = Dense(self.num_classes, activation='softmax')(output_img)
        
        return Model(d0, output_classes)
    
    def vanilla3DUNet(self):
        """U-Net for fully labeled data """

        
        # Image input
        d0 = Input(shape=self.img_shape)
        #dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv3d(d0, self.gf,isPool=False) #Size=512x512
        d2 = self.conv3d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv3d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv3d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv3d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv3d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv3d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv3d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv3d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv3d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv3D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        if self.num_classes < 2:
            output_classes = Dense(self.num_classes, activation='sigmoid')(output_img)
        else:
            output_classes = Dense(self.num_classes, activation='softmax')(output_img)
        
        return Model(d0, output_classes)
    
    def WNet(self):
        """W-Net for fully labeled data 
        
        Input: single or multi-sequence imaging data
        Outputs:    1. Classification labels
                    2. Reconstructed input from classification labels
        
        """

        
        # Image input
        d0 = Input(shape=self.img_shape)
        #dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv2d(d0, self.gf,isPool=False) #Size=512x512
        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv2D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        
        output_classes = Dense(self.num_classes, activation='softmax')(output_img)
        
        d1 = self.conv2d(output_classes, self.gf,isPool=False) #Size=512x512
#        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
#        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
#        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
#        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
#        
#        #Bridge 
#        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
#        
#        # Upsampling
#        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
#        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
#        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
#        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv2D(self.num_channels, kernel_size=3, strides=1, padding='same')(d1)
        
        return Model(d0, [output_classes,output_img])
    
    def S4Net_3D(self):
        """
        S4-Net: Self-supervised synthesis (suppression) using segmentation
        
        Input: single or multi-sequence imaging data
        Outputs:    1. Synthetic data based on previous segmentation (using another algorithm)
                    2. Reconstructed input from synthetic data
        
        """

        
        # Image input
        d0 = Input(shape=self.img_shape)
        #dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv3d(d0, self.gf,isPool=False) #Size=512x512
        d2 = self.conv3d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv3d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv3d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv3d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv3d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv3d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv3d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv3d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv3d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv3D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        
        output_classes = Dense(self.num_classes, activation='linear')(output_img)
        
        d1 = self.conv3d(output_classes, self.gf,isPool=False) #Size=512x512
        d2 = self.conv3d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv3d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv3d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv3d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv3d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv3d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv3d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv3d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv3d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv3D(self.num_channels, kernel_size=3, strides=1, padding='same')(u6)
        
        return Model(d0, [output_classes,output_img])
    
    def S4Net(self):
        """
        S4-Net: Self-supervised synthesis (suppression) using segmentation
        
        Input: single or multi-sequence imaging data
        Outputs:    1. Synthetic data based on previous segmentation (using another algorithm)
                    2. Reconstructed input from synthetic data
        
        """

        
        # Image input
        d0 = Input(shape=self.img_shape)
        #dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv2d(d0, self.gf,isPool=False) #Size=512x512
        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv2D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        
        output_classes = Dense(self.num_classes, activation='linear')(output_img)
        
        d1 = self.conv2d(output_classes, self.gf,isPool=False) #Size=512x512
        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv2D(self.num_channels, kernel_size=3, strides=1, padding='same')(u6)
        
        return Model(d0, [output_classes,output_img])
    
    def sparseUNet(self):
        """U-Net for sparsely labeled data """

        
        # Image input
        d0 = Input(shape=self.img_shape)
        #dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv2d(d0, self.gf,isPool=False) #Size=512x512
        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
        #d6 = conv2d(d5, self.gf*8)
        #d7 = conv2d(d6, self.gf*8)

        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
        #m2 = midConv2d(m1,self.gf*8)
        # Upsampling
        #u1 = deconv2d(d7, d6, self.gf*8)
        #u2 = deconv2d(m1, d5, self.gf*8) #
        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        #u7 = UpSampling2D(size=2)(u6)
        
        output_img = Conv2D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        
        output_classes = Dense(self.num_classes, activation='sigmoid')(output_img)
        
        lab = Input(shape=self.label_shape)
        output_test = Multiply()([lab, output_classes])
        

        return Model([d0,lab], output_test)
    
   
    def unsupervedUNet(self):
        """U-Net for fully labeled data """

        def func(x):

            greater = K.greater_equal(x, 0.5) #will return boolean values
            greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
            return greater 
        # Image input
        d0 = Input(shape=self.img_shape)
        dn = Dropout(0.5)(d0)

        # Downsampling
        d1 = self.conv2d(dn, self.gf,isPool=False) #Size=512x512
        d2 = self.conv2d(d1, self.gf*2) #Size = 256x256
        d3 = self.conv2d(d2, self.gf*4) # Size = 128x128
        d4 = self.conv2d(d3, self.gf*8) # Size = 64x64
        d5 = self.conv2d(d4, self.gf*16) #Size = 32x32
        
        #Bridge 
        m1 = self.midConv2d(d5,self.gf*16) #Size = 32x32
        
        # Upsampling
        u3 = self.deconv2d(m1, d4, self.gf*8) # Size = 64x64
        u4 = self.deconv2d(u3, d3, self.gf*4) # Size = 128x128
        u5 = self.deconv2d(u4, d2, self.gf*2) # Size = 256x256
        u6 = self.deconv2d(u5, d1, self.gf) # Size = 512x512

        output_img = Conv2D(self.num_classes, kernel_size=3, strides=1, padding='same', activation='relu')(u6)
        
        output_classes = Dense(self.num_classes, activation='sigmoid')(output_img)
        
        reconstructInput = Conv2D(self.num_channels, kernel_size=3, strides=1, padding='same')(output_classes) 
        
        return Model(d0, reconstructInput)
    
    def conv2d(self,layer_input, filters, f_size=3, bn=True, isPool=True):
            """Layers used during downsampling"""
            if isPool==True:
                d = MaxPooling2D(pool_size=(2,2))(layer_input)
            else:
                d = layer_input
                
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(d)            
            d = BatchNormalization(momentum=0.99)(d)            
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(d)
            #d = LeakyReLU(alpha=0.0)(d)
            d = BatchNormalization(momentum=0.99)(d)
            
  
            return d
        
    def deconv2d(self,layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Concatenate()([u, skip_input])
            
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            #u = LeakyReLU(alpha=0.0)(u)
            u = BatchNormalization(momentum=0.99)(u)
            
            
            
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            #u = LeakyReLU(alpha=0.0)(u)
            u = BatchNormalization(momentum=0.99)(u)
            

            return u
        
    def midConv2d(self,layer_input, filters, f_size=3, bn=True):
            """Layers used at the bridge"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(layer_input)
            
            d = BatchNormalization(momentum=0.99)(d)
            
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(d)
            
            d = BatchNormalization(momentum=0.99)(d)
            
            
            return d
        
    def conv3d(self,layer_input, filters, f_size=3, bn=True, isPool=True):
            """Layers used during downsampling"""
            if isPool==True:
                d = MaxPooling3D(pool_size=(2,2,2))(layer_input)
            else:
                d = layer_input
                
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(d)            
            d = BatchNormalization(momentum=0.99)(d)            
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(d)
            #d = LeakyReLU(alpha=0.0)(d)
            d = BatchNormalization(momentum=0.99)(d)
            
  
            return d
        
    def deconv3d(self,layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling3D(size=2)(layer_input)
            u = Concatenate()([u, skip_input])
            
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            #u = LeakyReLU(alpha=0.0)(u)
            u = BatchNormalization(momentum=0.99)(u)
            
            
            
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            #u = LeakyReLU(alpha=0.0)(u)
            u = BatchNormalization(momentum=0.99)(u)
            

            return u
        
    def midConv3d(self,layer_input, filters, f_size=3, bn=True):
            """Layers used at the bridge"""
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(layer_input)
            
            d = BatchNormalization(momentum=0.99)(d)
            
            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(d)
            
            d = BatchNormalization(momentum=0.99)(d)
            
            
            return d