# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:46:30 2019

@author: Raghav
"""

from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import functools

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
              'Z']


_model = load_model('./Models/CNN/CNN3.h5')
classification_result = []
files = []
for filename in os.listdir('./output/segments/'):
    files.append(filename)

def sorted_by(a,b):
    val1 = int(a.split('.')[0])
    val2 = int(b.split('.')[0])
    
    print((val1,val2))
    
    if(val1 < val2):
        return -1
    elif(val1 > val2):
        return 1
    
    return 0


cmp = functools.cmp_to_key(sorted_by)
files.sort(key=cmp)

for filename in files:
    segment = image.load_img('output/segments/' + filename)
    img_tensor = image.img_to_array(segment)                  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.
    
    class_ = _model.predict(img_tensor)
    index = np.argmax(class_)
    output = CHARACTERS[index]
    
    classification_result.append(output)

plate_string = ''
for pred in classification_result:
    plate_string += pred[0]

print(plate_string)