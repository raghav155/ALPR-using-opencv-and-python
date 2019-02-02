# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:05:15 2019

@author: Raghav
"""

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Dropout

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
              'Z']

model = Sequential()

# first layer conv->maxpool
model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# second layer conv->maxpool
model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

'''
# third layer conv->maxpool
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
'''

# Flatten Layer
model.add(Flatten())

# Full connection Layer
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))

# Output layer
model.add(Dense(units=36, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# Augmentation of data
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'training_data/',
    target_size=(20, 20),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'test_data/',
    target_size=(20, 20),
    batch_size=32,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / 32, 
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / 32)

model.evaluate_generator(validation_generator, steps=32)
model.save('../Models/CNN/CNN3.h5')

import matplotlib.pyplot as plt

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()

# Single Prediction
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

_model = load_model('../Models/CNN/CNN2.h5')
img = image.load_img('../test_image.png')

# converting image to a tensor
img_tensor = image.img_to_array(img)                  # (height, width, channels)
img_tensor = np.expand_dims(img_tensor, axis=0)         
img_tensor /= 255.

class_ = _model.predict(img_tensor)
index = np.argmax(class_)
output = CHARACTERS[index]

print(output)