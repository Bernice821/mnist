# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:06:54 2021

@author: reta
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:15:06 2021

@author: shu
"""
import os
modelPath = os.path.join(".", "model")
mypath= os.path.join(".", "test_num")
outPath = os.path.join(".", "output")
import matplotlib.pyplot as plt #display image
#from datetime import tf
from tensorflow.keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import mnist #load data
from keras import models #train model architecture
from keras import layers #train model architecture
from keras.utils import to_categorical #label to categorical type輔助工具
from keras.models import load_model

from os import listdir 
from os.path import isfile, join 
import numpy as np
import cv2 

((train_images,train_labels),(test_images,test_labels))=mnist.load_data()
#training data
print("Training image numDimesion/Rank-->")
print(train_images.ndim) #3
print("Training image shape-->")
print(train_images.shape)
print("Training label shape-->")
print("Training image datatype-->",type(train_images[0,0,0]))
print(train_images.dtype)
print(train_labels.shape)
print("Training data length-->")
print(len(train_labels))

#testing data
print("Training image shape-->")
print(test_images.shape)
print("Testing label shape-->")
print(test_labels.shape)
print("Testing data length-->")
print(len(test_labels))


#data slicing
slice1=train_images[10:100]
slice2=train_images[10:100,:,:]
slice3=train_images[10:100,0:28,0:28]
slice4=train_images[10,14:28,14:28]
slice4=train_images[10,7:-7]

#Training Model architecture
trainedModel=models.Sequential()
trainedModel.add(layers.Dense(512,name="Dense_1",activation='relu',input_shape=(28*28,)))
trainedModel.add(layers.Dense(512,name="Dense_2",activation='relu',input_shape=(28*28,)))
trainedModel.add(layers.Dense(10,activation="softmax"))
print(trainedModel.summary())
 
trainedModel.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

#Data Processing
train_images=train_images.reshape((60000,28*28))
print("0~255 integer:",train_images[0][:])
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
train_labels=to_categorical(train_labels)
test_labels_original=test_labels
test_labels=to_categorical(test_labels)


#模型訓練中
trainedModel.fit(train_images,train_labels,epochs=5,batch_size=128)

trainedModel=load_model("model_mnist_keras_0.h5")


#Save Model
modelName="model_mnist_keras_0.h5"
trainedModel.save(modelName)
del trainedModel
trainedModel=load_model(modelName)
print("model summary ",trainedModel.summary())


testLoss,testAcc=trainedModel.evaluate(test_images,test_labels)
print("testLoss ",testLoss)
print("testAcc ",testAcc)


#Make prediction
#換test_images at least 10 images
#test_labels要自己給
#用imageJ傳換成灰階，壓縮成28*28

#img = cv2.imread('./test_num/0.png')
test_images =[]
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] 
images = np.empty(len(onlyfiles), dtype=object) 
for n in range(0, len(onlyfiles)): 
    images[n] = cv2.imread(join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE) 
    images[n] = cv2.resize(images[n], (28, 28))
    images[n] = 255 - images[n]
    test_images.append(images[n])
test_images =np.array(test_images)    

print(test_images.shape)

test_images = test_images.reshape((10, 28 * 28))
test_images = test_images.astype('float32') / 255


test_labels = [0,1,2,3,4,5,6,7,8,9]
test_labels = to_categorical(test_labels)
pc=trainedModel.predict_classes(test_images)
ps=trainedModel.predict(test_images)
ps.round(1)
print("Class of prediction:",pc[0:10])
print("Result of prediction:",ps)
print("Label of testing:",test_labels)

#Plot output
saveFileName="image_0.jpg"
#plotEvaluateImage(test_images_original,test_labels_original,pc,n)
plt.figure(figsize=(15,10))
plt.title("Classification")
for ii in range(5):
    ax=plt.subplot(2,5,1+ii)
    ax.imshow(test_images[ii].reshape(28,28),cmap='binary') #binary黑白交換
    ax.set_title("label={}\npredi={}".format(str(test_labels[ii]),str(pc[ii],fontsize=15)))
    ax.set_xticks([])
    ax.set_yticks([])
    
for ii in range(5): #5~10
    ax=plt.subplot(2,5,6+ii)
    ax.imshow(test_images[ii+5].reshape(28,28),cmap='binary') #binary黑白交換
    ax.set_title("label={}\npredi={}".format(str(test_labels[ii+5]),str(pc[ii+5],fontsize=15)))
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig("MNIST.jpg")
plt.show()



 