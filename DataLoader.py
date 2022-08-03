# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:11:31 2021

@author: vishwa
"""
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.utils import to_categorical
from skimage.color import rgb2gray
import pydicom
from PIL import Image 
import cv2
import pandas as pd
import os 

class DataLoader:
    def __init__(self, data_path,data_type='tissue_signature_mat',task_type='segmentation',csv_path=None):
        
        self.path = data_path
        self.load = self.get_loader(data_type,task_type)   
        self.csv_path = csv_path
        
    def get_loader(self,data_type,task_type):
        allLoaders = {'tissue_signature_mat':{'segmentation':self.load_data_mat_seg}, 'train_labels_mat':{'segmentation':self.load_data_mat_seg2},'classification3D_mat':{'classification':self.load_mat_classification3D},'classification3D_dicom':{'classification':self.load_dicom_classification3D},'image_synthesis_mat':{'synthesis':self.load_data_mat_image_synthesis},'dicom_folder':{'classification':self.load_folderwise_classifications}}
        return allLoaders[data_type][task_type]
    
    def load_data_mat_seg(self,batch_size=1):
        '''
        This will load segmentation data stored in mat files. The format for
        each mat files is as follows:
        1. train_x: flattened input data 
        2. AllLabels1: flattened segmentation labels
        3. imSize_forDN: reshaping size for the input image and labels
        '''
        # This will get all the files in the path
        path = glob('%s/*' %(self.path))
        # Randomly shuffle all the files
        np.random.shuffle(path)
        n_batches = int(len(path)/batch_size)
        self.n_batches = n_batches
        for i in range(n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs = []
            AllLabels = []
            for batchFile in batch:
                img, AllLabels1, imSize = self.read_tissue_signature(batchFile) # Decide what to load
                
                img = img.reshape(imSize)
                imgs.append(img)
                
                imShape = [imSize[0],imSize[1],np.shape(AllLabels1)[1]]
                AllLabels1 = AllLabels1.reshape(imShape)
                AllLabels.append(AllLabels1)

            imgs = np.array(imgs)
            AllLabels = np.array(AllLabels)

            yield imgs, AllLabels, batch
    
    def load_data_mat_seg2(self,batch_size=1):
        '''
        This will load segmentation data stored in the following format in mat files
        1. inputData: input Image
        2. outputMask: Corresponding segmentation label
        '''
        # This will get all the files in the path
        path = glob('%s/*' %(self.path))
        # Randomly shuffle all the files
        np.random.shuffle(path)
        n_batches = int(len(path)/batch_size)
        self.n_batches = n_batches
        for i in range(n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs = []
            AllLabels = []
            for batchFile in batch:
                img, AllLabels1, imSize = self.read_train_labels(batchFile) # Decide what to load
                #img = np.expand_dims(img,-1)    # Comment for brain breast experiment.                    
                imgs.append(img)                
                AllLabels.append(AllLabels1)

            imgs = np.array(imgs)
            AllLabels = np.array(AllLabels)
            
            yield imgs, AllLabels, batch
    
    def load_dicom_classification3D(self,batch_size=1):
        '''
        This will load classification data. The folder should contain the following

        1. CSV file (name should be classificationData.csv) with the following column headings
            img: name of the image dicom file
            class: corresponding classification label
        '''
        # Read the csv file
        df = pd.read_csv(self.path + '/classificationData.csv')
        # Get all the images and class labels
        allImgs = df['patientId']
        allLabels = df['Sex']
        
        # Create a shuffled index for training
        ixImgs = np.arange(len(allImgs))        
        np.random.shuffle(ixImgs)
        
        # Randomly shuffle all the files        
        n_batches = int(len(ixImgs)/batch_size)
        self.n_batches = n_batches
        for i in range(n_batches):
            ind = ixImgs[i*batch_size:(i+1)*batch_size]
            batch = allImgs[ind]
            classLabels = allLabels[ind]
            imgs = []
            for batchFile in batch:
                img,imSize = self.read_classification_dicom(self.path + '/' + batchFile) # Decide what to load                
                img = np.expand_dims(img,-1)
                imgs.append(img)                                
                    
            imgs = np.array(imgs)
            AllLabels = np.array(classLabels)

            yield imgs, AllLabels, batch
            
    def load_mat_classification3D(self,batch_size=1):
        '''
        This will load classification data. The folder should contain the following
        1. Mat files in the following format
            img: input Image
        2. CSV file (name should be classificationData.csv) with the following column headings
            img: name of the image mat file
            class: corresponding classification label
        '''
        # Read the csv file
        df = pd.read_csv(self.path + '/classificationData.csv')
        # Get all the images and class labels
        allImgs = df['img']
        allLabels = df['class']
        
        # Create a shuffled index for training
        ixImgs = np.arange(len(allImgs))        
        np.random.shuffle(ixImgs)
        
        # Randomly shuffle all the files        
        n_batches = int(len(ixImgs)/batch_size)
        self.n_batches = n_batches
        for i in range(n_batches):
            ind = ixImgs[i*batch_size:(i+1)*batch_size]
            batch = allImgs[ind]
            classLabels = allLabels[ind]
            imgs = []
            for batchFile in batch:
                img,imSize = self.read_classification_mat(self.path + '/' + batchFile) # Decide what to load                
                img = np.expand_dims(img,-1)
                imgs.append(img)                                
                    
            imgs = np.array(imgs)
            AllLabels = np.array(classLabels)

            yield imgs, AllLabels, batch
    
    def load_data_mat_image_synthesis(self,batch_size=1):
        '''
        This will load image synthesis data. The folder should contain the following
        Mat files in the following format
            imgIn: input Image
            imgOut: output Image
        
        '''
        
        # This will get all the files in the path
        path = glob('%s/*' %(self.path))
        # Randomly shuffle all the files
        np.random.shuffle(path)
        n_batches = int(len(path)/batch_size)
        self.n_batches = n_batches
        for i in range(n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_In = []
            imgs_Out = []
            for batchFile in batch:
                try:
                    imgIn, imgOut, imSize = self.read_synthesis_data(batchFile) # Decide what to load
                    #print(imgIn.shape)
                    #print(imgOut.shape)
                    imgIn = cv2.resize(imgIn, dsize=(256,256))
                    imgOut = cv2.resize(imgOut, dsize=(256,256))
                    imgIn = np.expand_dims(imgIn,-1)
                    imgOut = np.expand_dims(imgOut,-1)
                    imgs_In.append(imgIn)                
                    imgs_Out.append(imgOut)                
                except:
                    print('Doing nothing.. skipping')

            imgs_In = np.array(imgs_In)
            imgs_Out = np.array(imgs_Out)
            

            yield imgs_In, imgs_Out, batch
    
    def load_folderwise_classifications(self,batch_size=1):
        '''
        This will load image classification data. The folder should subfolders
        such that every file in the subfolder would have the same class. The data
        should be accompanied with an excel sheet with each subfolder class
        '''
        
        # Read the csv file
        df = pd.read_csv(self.csv_path)
        ids = df['id']
        class_columns = df.columns
        class_columns = (class_columns.drop('id'))
        df_classes = df[class_columns]
        classArray = df_classes.to_numpy()
        
        n_batches = int(len(ids)/batch_size)
        self.n_batches = n_batches
        for i in range(n_batches-1):
            batch = ids[i*batch_size:(i+1)*batch_size]
            imgs_In = []
            imgs_Out = []
            os.chdir(self.path)
            for iter1,batchFolder in enumerate(batch):
                for root, dirs, files in os.walk(batchFolder):
                    for file in files:
                        if file.endswith(".mat"):
                            #dataset = pydicom.dcmread(os.path.join(root,file))
                            #img = np.array(dataset.pixel_array)
                            img,imSize = self.read_classification_mat(os.path.join(root,file))
                            #print(img.shape)
                            imgIn = cv2.resize(img, dsize=(512,512))
                            #imSize = [512,512,1]
                            #imgIn = (img-np.min(img))/(np.max(img)-np.min(img)) 
                            imgIn = np.expand_dims(imgIn,-1)
                            imgs_In.append(imgIn)  
                            imgs_Out.append(classArray[i*batch_size+iter1,:])
            
            imgs_In = np.array(imgs_In)
            imgs_Out = np.array(imgs_Out)
            yield imgs_In, np.array(imgs_Out), batch
        
    
    def read_tissue_signature(self, path):
        
        readmat = scipy.io.loadmat(path)
        img = np.array(readmat.get('train_x'))        
        labels = np.array(readmat.get('AllLabels1'))
        imSize = np.array(readmat.get('imSize_forDN'))
        return img,labels,imSize
    
    def read_train_labels(self, path):
        
        readmat = scipy.io.loadmat(path)
        #img = np.array(readmat.get('inputData'))  
        img = np.array(readmat.get('img'))  
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = cv2.resize(img, dsize=(256,256))
        #labels_full = (readmat.get('outputMask'))
        labels_full = (readmat.get('mask'))
        labels_full = cv2.resize(labels_full, dsize=(256,256))
        #labels_full = labels_full.astype(np.int)
        
        if np.max(labels_full)>1:
            labels = to_categorical(labels_full,np.max(labels_full)+1)
        else:
            labels = np.expand_dims(labels_full,-1)            
        imSize = np.shape(img)
        
        return img,labels,imSize
    
    def read_synthesis_data(self, path):
        
        readmat = scipy.io.loadmat(path)
        #img = np.array(readmat.get('imgIn'))
        img = np.array(readmat.get('PD_Img'))
        imgIn = (img-np.min(img))/(np.max(img)-np.min(img))
        
        #img = np.array(readmat.get('imgOut'))  
        img = np.array(readmat.get('PD_Suppressed'))  
        imgOut = (img-np.min(img))/(np.max(img)-np.min(img))
        
        imSize = np.shape(img)
        
        return imgIn,imgOut,imSize
    
    def read_classification_mat(self, path):
        
        readmat = scipy.io.loadmat(path)
        img = np.array(readmat.get('img'))  
        #print(img.shape)
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        imSize = np.shape(img)
        
        return img,imSize
    
    def read_classification_dicom(self, path):
        
        dataset = pydicom.dcmread(os.path.join(path+'.dcm'))
        img = np.array(dataset.pixel_array)
         
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = cv2.resize(img, dsize=(256,256))
        imSize = np.shape(img)
        
        return img,imSize
                
