# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:26:10 2019

@author: Raghav
"""

# Importing the keras libraries 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import numpy as np

#Intialising the CNN
classifier = Sequential()

# Step-1 - Convolutional
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape = (20, 20, 1), activation = 'relu'))
classifier.add(Convolution2D(filters=64, kernel_size=(3,3), activation='relu'))
# Step-2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# adding one more convolutional layer
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step-3 - Flattening
classifier.add(Flatten())

# Step-4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 36, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
              'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

DICT = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11,
        'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20, 'V':21,
        'W':22, 'X':23, 'Y':24, 'Z':25, '1':26, '2':27, '3':28, '4':29, '5':30, '6':31, '7':32, '8':33, '9':34, '0':35}

DIRECTORY = './Dataset/'

def trainingDataFormation():
    image_data = []
    target_value = []
    
    for fname in range(27,63):
        currentDirectory = DIRECTORY + str(fname) + '/'
        for cfile in range(1,31):
            currentFile = currentDirectory + str(cfile) + '_20by20.png'
            cimage = image.load_img(path=currentFile, color_mode='grayscale')
            cimage = image.img_to_array(cimage)
            image_data.append(cimage)
            target_value.append(DICT[CHARACTERS[fname-1]])
        
    return (np.array(image_data), np.array(target_value))


import keras
image_data, target_value = trainingDataFormation()
one_hot_labels = keras.utils.to_categorical(target_value, num_classes=36)
image_data = image_data.astype(np.float32)
image_data = image_data/255

classifier.fit(image_data, one_hot_labels, epochs=100)
test_image = image.load_img('test_image.png',color_mode='grayscale', target_size=(20,20))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

index = np.argmax(result[0])
print(CHARACTERS[index+26])

# Save model
cnn_model = '.\\Models\\CNN\\CNN.h5'
classifier.save(cnn_model)