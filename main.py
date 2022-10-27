import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import skimage.io as io
import os

import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize, to_categorical
from keras.layers import Conv2D, Dropout, Flatten, Activation, MaxPooling2D, Dense
from keras.models import Sequential

import cv2 as cv
from skimage import data, color
from skimage.transform import rescale, resize


from PIL import Image
dataset = []
label = []
path = 'C:\\Users\\SAR Computers\\Documents\\datasets\\meni'
meni_tumor_images = os.listdir(path)

for image_name in meni_tumor_images:
    img = io.imread(os.path.join(path, image_name), as_gray=True)
    img = resize(img, (200,200),
                       anti_aliasing=True)
    dataset.append(np.array(img))
    label.append("Meningioma")
    
    path_two = 'C:\\Users\\SAR Computers\\Documents\\datasets\\no'
no_tumor_images = os.listdir(path_two)

for image_name in no_tumor_images:
    img = io.imread(os.path.join(path_two, image_name), as_gray=True)
    img = resize(img, (200,200),
                       anti_aliasing=True)
    dataset.append(np.array(img))
    label.append("No tumor")
    
    path_three = 'C:\\Users\\SAR Computers\\Documents\\datasets\\gli'
gli_tumor_images = os.listdir(path_three)

for image_name in gli_tumor_images:
    img = io.imread(os.path.join(path_three, image_name), as_gray=True)
    img = resize(img, (200,200),
                       anti_aliasing=True)
    dataset.append(np.array(img))
    label.append("Glioma")
    
    
path_four = 'C:\\Users\\SAR Computers\\Documents\\datasets\\piti'
piti_tumor_images = os.listdir(path_four)

for image_name in piti_tumor_images:
    img = io.imread(os.path.join(path_four, image_name), as_gray=True)
    img = resize(img, (200,200),
                       anti_aliasing=True)
    dataset.append(np.array(img))
    label.append("Pitiuary")
    
dataset = np.array(dataset)
label = np.array(label)


le = LabelEncoder()
label = le.fit_transform(label)
label

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.25, random_state=42)

X_train = normalize(X_train)
X_test = normalize(X_test)
X_train
#to categorical label


model = Sequential()

model.add(Conv2D(filters= 16,kernel_size= (3,3), input_shape=(200,200,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
​
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
​
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
​
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
​
model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=10,verbose=1,validation_data=(X_test, y_test),shuffle=False)

print("Generate predictions for the 3 samples from ismail")
predictions = model.predict(yes_testing[:6])
predictions
