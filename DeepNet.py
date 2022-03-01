# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:01:57 2021

@author: vishwa
"""
from UNet import UNet
from Classifier_3D import Classifier_3D
from DataLoader import DataLoader
import datetime
import numpy as np
#from skimage.io import imsave
import os
from tensorflow.keras.models import Model
from matplotlib.pyplot import imsave
import pandas as pd
class DeepNet:
    def __init__(self, data_path=None, 
                 data_type='tissue_signature_mat', 
                 task_type='segmentation', 
                 model_type='unet',
                 model_subtype='vanilla',
                 imSize = [512,512,1],
                 labelSize = None,
                 csv_path = None):
        
        modelObject, self.train, self.test = self.getModel(model_type,model_subtype,imSize,labelSize)
        self.model = modelObject.model
        self.data_loader = DataLoader(data_path,data_type,task_type,csv_path)
        self.test_data_loader = DataLoader(data_path,data_type,task_type,csv_path)
        self.imSize = imSize
        self.labelSize = labelSize
        self.model_type = model_type
        self.model_subtype = model_subtype
        self.task_type = task_type
        self.data_type=  data_type
    
    def getModel(self,model_type,model_subtype,imSize,labelSize):
        
        if model_type == 'unet':
            allModels = UNet(model_subtype,imSize,labelSize)
        elif model_type == 'classifier3D':
            allModels = Classifier_3D(model_subtype,imSize,labelSize)
        #allModels = {'unet': UNet(model_subtype,imSize,labelSize),'classifier3D':Classifier_3D(model_subtype,imSize,labelSize)}
        allTrainers = {'unet': {'vanilla':self.train_vanilla, 'vanilla3D':self.train_vanilla, 'unsupervised':self.train_unsupervised, 'WNet':self.train_WNet, 'S4Net':self.train_WNet},'classifier3D':{'vanilla3D':self.train_vanilla,'vanilla2D':self.train_vanilla}}
        allTesters = {'unet': {'vanilla':self.test_vanilla, 'vanilla3D':self.test_vanilla3D, 'unsupervised':self.test_unsupervised,'WNet':self.test_WNet,'S4Net':self.test_S4Net},'classifier3D':{'vanilla3D':self.test_vanilla_classifier3D,'vanilla2D':self.test_vanilla_classifier3D}}
        return allModels, allTrainers[model_type][model_subtype], allTesters[model_type][model_subtype]
        
    def train_vanilla(self,epochs=10,batch_size=10):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs, AllLabels,fName) in enumerate(self.data_loader.load(batch_size)):

                #print(imgs.shape)
                #print(AllLabels.shape)
                g_loss = self.model.train_on_batch(imgs,AllLabels)
                
                
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        g_loss[0], 100*g_loss[1],
                                                                        elapsed_time))
            
            #self.model.save('C:/Users/Adv Clinical Breast/OneDrive - Johns Hopkins/MyPapers/COVID.AI/Trained_3DClassifer_v1.h5')
            
            
     
    def test_vanilla_classifier3D(self,path_testData=None,path_saveResults=None):
        

        if path_testData is not None:
            self.test_data_loader.path = path_testData
        pred = []
        gt = []
        allName = []
        for batch_i, (imgs, AllLabels,fName) in enumerate(self.test_data_loader.load(batch_size=1)):

            #lab_mask2 = np.max(AllLabels,3)
            #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
            #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
            
            #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
            #print(imgs.shape)
            pred.append(self.model.predict(imgs))
            gt.append(AllLabels)
            allName.append(fName)
        df = pd.DataFrame({'Patient':allName,'gt':gt,'pred':pred})
        df.to_csv(path_saveResults + 'output_2D_XRAY_Classifier.csv')
        
              
    def test_vanilla(self,path_testData=None,path_saveResults=None):
        

        if path_testData is not None:
            self.test_data_loader.path = path_testData
        for batch_i, (imgs, AllLabels,fName) in enumerate(self.test_data_loader.load(batch_size=1)):

            #lab_mask2 = np.max(AllLabels,3)
            #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
            #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
            
            #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
            print(imgs.shape)
            predImage = self.model.predict(imgs)
            img = imgs[0,:,:,0]
            print(img.shape)
            imSize = predImage.shape
            predImage = predImage.reshape(imSize[1],imSize[2])
            print(predImage.shape)
            #predImage = np.argmax(predImage,2)
            print(predImage.shape)
            imsave(os.path.join(path_saveResults + str(batch_i) + '_origMask.png'),AllLabels[0,:,:,0],cmap='gray')
            imsave(os.path.join(path_saveResults + str(batch_i) + '_UNet.png'),predImage,cmap='gray')
            imsave(os.path.join(path_saveResults + str(batch_i) + '_origImg.png'),img,cmap='gray')
            
    
    def test_vanilla3D(self,path_testData=None,path_saveResults=None):
        

        if path_testData is not None:
            self.test_data_loader.path = path_testData
        for batch_i, (imgs, AllLabels,fName) in enumerate(self.test_data_loader.load(batch_size=1)):

            #lab_mask2 = np.max(AllLabels,3)
            #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
            #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
            
            #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
            print(imgs.shape)
            predImage = self.model.predict(imgs)
            for i in range (0,imgs.shape[3]):
                img = imgs[0,:,:,i,0]
                print(img.shape)
                pred = predImage[0,:,:,i,0]
                imsave(os.path.join(path_saveResults + str(batch_i) + '_' + str(i) + '_UNet_Lung.png'),pred,cmap='jet')
                imsave(os.path.join(path_saveResults + str(batch_i) + '_' + str(i) + '_orig_Lung.png'),img,cmap='gray')
                
    
    def train_unsupervised(self,epochs=10,batch_size=10):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs, AllLabels,fName) in enumerate(self.data_loader.load(batch_size)):

                #lab_mask2 = np.max(AllLabels,3)
                #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
                #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
                
                #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
                g_loss = self.model.train_on_batch(imgs,imgs)
                
                
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        g_loss[0], 100*g_loss[1],
                                                                        elapsed_time))
                
                    
    def test_unsupervised(self,path_testData=None,path_saveResults=None):
        
        intermediate_layer_model = Model(inputs=self.model.input,
                                 outputs=self.model.layers[-2].output)
        
        print(self.model.layers[-2].output)
        if path_testData is not None:
            self.test_data_loader.path = path_testData
        for batch_i, (imgs, AllLabels,fName) in enumerate(self.test_data_loader.load(batch_size=1)):

            #lab_mask2 = np.max(AllLabels,3)
            #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
            #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
            
            #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
            print(imgs.shape)
            predImage = intermediate_layer_model.predict(imgs)
            img = imgs[0,:,:,0]
            print(img.shape)
            imSize = predImage.shape
            predImage = predImage.reshape(imSize[1],imSize[2],imSize[3])
            print(predImage.shape)
            #predImage = np.argmax(predImage,2)
            #print(predImage.shape)
            for i in range(0,imSize[3]):
                reconImg = predImage[:,:,i]
                imsave(os.path.join(path_saveResults + str(batch_i) + '_UNet_MD' + str(i) + '.png'),reconImg,cmap='jet')
                
            imsave(os.path.join(path_saveResults + str(batch_i) + '_origF_MD.png'),img,cmap='gray')
            
            predImage = self.model.predict(imgs)
            imSize = predImage.shape
            predImage = predImage.reshape(imSize[1],imSize[2],imSize[3])            
            for i in range(0,imSize[3]):
                reconImg = predImage[:,:,i]
                imsave(os.path.join(path_saveResults + str(batch_i) + '_recon_MD' + str(i) + '.png'),reconImg,cmap='gray')
            
      
    def train_WNet(self,epochs=10,batch_size=10):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs, AllLabels,fName) in enumerate(self.data_loader.load(batch_size)):

                #lab_mask2 = np.max(AllLabels,3)
                #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
                #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
                
                #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
                #print(imgs.shape)
                #print(AllLabels.shape)
                try:
                    g_loss = self.model.train_on_batch(imgs,[AllLabels,imgs])
                    
                    
                    elapsed_time = datetime.datetime.now() - start_time
                    # Plot the progress
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            g_loss[0], 100*g_loss[1],elapsed_time))
                except: 
                    print('No data in this iteration - moving on')
                
    def test_WNet(self,path_testData=None,path_saveResults=None):
        
        
        if path_testData is not None:
            self.test_data_loader.path = path_testData
        
        allData = {};
        allData['BG']= []
        allData['Muscle'] = [];
        allData['Bone']=[];
        allData['Fat'] = [];
        allData['FatInf'] = [];
        allData['totalFat'] = [];
        allData['fName'] = [];
        for batch_i, (imgs, AllLabels,fName) in enumerate(self.test_data_loader.load(batch_size=1)):
            #if batch_i>=20:
            #    break

            #lab_mask2 = np.max(AllLabels,3)
            #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
            #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
            
            #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
            fName = fName[0]
            ind1 = fName.find('\\')
            fName = fName[ind1+1:]
            #print(imgs.shape)
            predImage,reconImgs = self.model.predict(imgs)
            img = imgs[0,:,:,0]
            #print(img.shape)
            imSize = predImage.shape
            predImage = predImage.reshape(imSize[1],imSize[2],imSize[3])
            #print(predImage.shape)
            predImage = np.argmax(predImage,2)
            #print(predImage.shape)
            #print(np.sum(predImage==0))
            allData['BG'].append(np.sum(predImage==0)+np.sum(predImage==1));
            allData['Muscle'].append(np.sum(predImage==2))
            allData['Bone'].append(np.sum(predImage==3))
            allData['Fat'].append(np.sum(predImage==4))
            allData['FatInf'].append(np.sum(predImage==5))
            allData['totalFat'].append(np.sum(predImage==5)+np.sum(predImage==4))
            allData['fName'].append(fName)
            
            imsave(os.path.join(path_saveResults + fName + '_UNet_MD.png'),predImage,cmap='jet')
            
            #predImage = np.argmax(predImage,2)
            #print(predImage.shape)
            #for i in range(0,imSize[3]):
            #    reconImg = predImage[:,:,i]
            #    imsave(os.path.join(path_saveResults + str(batch_i) + '_UNet_MD' + str(i) + '.png'),reconImg,cmap='jet')
                
            #imsave(os.path.join(path_saveResults + fName + '_origF_MD.png'),img,cmap='gray')
            
            predImage = reconImgs
            imSize = predImage.shape
            predImage = predImage.reshape(imSize[1],imSize[2],imSize[3])            
            for i in range(0,imSize[3]):
                reconImg = predImage[:,:,i]
                imsave(os.path.join(path_saveResults + fName + '_recon_MD' + str(i) + '.png'),reconImg,cmap='gray')                
                img = imgs[0,:,:,i]
                imsave(os.path.join(path_saveResults + fName + '_orig_MD' + str(i) + '.png'),img,cmap='gray')
        
            df = pd.DataFrame(allData)
            df.to_csv(path_saveResults + 'output_PixelNumbers.csv')
    def test_S4Net(self,path_testData=None,path_saveResults=None):
        
        
        if path_testData is not None:
            self.test_data_loader.path = path_testData
        for batch_i, (imgs, AllLabels,fName) in enumerate(self.test_data_loader.load(batch_size=1)):

            #lab_mask2 = np.max(AllLabels,3)
            #lab_mask2 = lab_mask2.reshape((lab_mask2.shape)[0],(lab_mask2.shape)[1],(lab_mask2.shape)[2],1)
            #lab_mask3 = np.repeat(lab_mask2,self.num_classes,3)
            
            #g_loss = self.unet.train_on_batch([imgs, lab_mask3],AllLabels)
            
            predImgs,reconImgs = self.model.predict(imgs)
            print(predImgs.shape,reconImgs.shape,imgs.shape)
            
            img = imgs[0,:,:,0]
            print(img.shape)
            #imSize = predImage.shape
            for i in range(predImgs.shape[3]):
                predImage = predImgs[0,:,:,i]
                reconImage = reconImgs[0,:,:,i]
                img = imgs[0,:,:,i]
                print(predImage.shape)
                
                
                
                imsave(os.path.join(path_saveResults + str(batch_i) + '_' + str(i) + '_orig_DCE.png'),img,cmap='gray')
                imsave(os.path.join(path_saveResults + str(batch_i) + '_' + str(i) + '_fatSuppressed_DCE.png'),predImage,cmap='gray')
                imsave(os.path.join(path_saveResults + str(batch_i) + '_' + str(i) + '_recon_DCE.png'),reconImage,cmap='gray')
                
#            predImage = reconImgs
#            imSize = predImage.shape
#            predImage = predImage.reshape(imSize[1],imSize[2],imSize[3])            
#            for i in range(0,imSize[3]):
#                reconImg = predImage[:,:,i]
#                imsave(os.path.join(path_saveResults + str(batch_i) + '_recon_DCE' + str(i) + '.png'),reconImg,cmap='gray')
