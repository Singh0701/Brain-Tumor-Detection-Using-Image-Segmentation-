# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:41:38 2021

@author: singh
"""

import os
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
 
encoder = OneHotEncoder()
encoder.fit([[0],[1]])      
# 0 - Normal
# 1 - Tumor


data = [] #List for storing the images in numpy array form
paths = [] #List for storing paths for all images
target_class = [] #To store the one hot encoded form of the image for the target class (Normal or Tumor)

for raw,dirc,folder in os.walk(r'C:\Users\singh\Downloads\arc\brain_tumor_dataset\yes'):
    for file in folder:
        if'.jpg' in file:
            paths.append(os.path.join(raw, file))
            
for path in paths:
    img = Image.open(path) #Opening the image file from path
    img = img.resize((128,128)) #Resizing the image 
    img = np.array(img) #Storing the image in form of numpy array
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        target_class.append(encoder.transform([[0]]).toarray()) #Assign target class as 0, tumor present
        
        
paths = []

for raw,dirc,folder in os.walk(r'C:\Users\singh\Downloads\arc\brain_tumor_dataset\no'):
    for file in folder:
        if'.jpg' in file:
            paths.append(os.path.join(raw, file))
            
for path in paths:
    img = Image.open(path) #Opening the image file from path
    img = img.resize((128,128)) #Resizing the image 
    img = np.array(img) #Storing the image in form of numpy array
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        target_class.append(encoder.transform([[1]]).toarray()) #Assign target class as 1, tumor not present


data = np.array(data) 
target_class = np.array(target_class)
target_class = target_class.reshape(139,2)

print(target_class.shape)


## Now Splitting the dataset into train and test in 80:20 ratio

x_train,x_test,y_train,y_test = train_test_split(data, target_class, test_size=0.2, shuffle=True, random_state=0)

print(x_train.shape)

model = Sequential()

model.add(Conv2D(32, kernel_size = (2,2), input_shape = (128,128,3), padding = 'Same'))

model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

amodel.compile(loss = "categorical_crossentropy", optimizer='Adamax')
print(model.summary())

print(y_train.shape)

history = model.fit(x_train, y_train, epochs = 30, batch_size = 40, verbose = 1,validation_data = (x_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()

print(model)

def names(number):
    if number==0:
        return 'It is a Tumor'
    else:
        return 'No, It is not a tumor'
    
from matplotlib.pyplot import imshow
img = Image.open(r"C:\Users\singh\Desktop\__results___20_1.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence that ' + names(classification))
img = Image.open(r"C:\Users\singh\Desktop\Y157.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence that ' + names(classification))