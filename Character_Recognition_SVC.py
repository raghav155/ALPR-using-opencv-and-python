# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:33:57 2019

@author: Raghav
"""

import numpy as np
import cv2
from sklearn.externals import joblib
import os
import functools


svc_classifier = joblib.load('./Models/SVC/svc.pkl')
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
    segment = cv2.imread('./output/segments/' + filename, 0)
    ret, letter_inverse_threshold = cv2.threshold(segment, 90, 255, cv2.THRESH_BINARY)
    letter_inverse_threshold = np.reshape(letter_inverse_threshold, (1,400))
    result = svc_classifier.predict(letter_inverse_threshold)
    classification_result.append(result)

plate_string = ''
for pred in classification_result:
    plate_string += pred[0]

print(plate_string)