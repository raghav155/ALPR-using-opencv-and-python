# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:56:59 2019

@author: Raghav
"""

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import os

dirs = os.listdir('output/')

DIRECTORY = './output/'

def trainingDataFormation():
    image_data = []
    target_value = []
    
    for dname in dirs:
        currentDirectory = DIRECTORY + str(dname) + '/'
        files = os.listdir(currentDirectory)
        for cfile in files:
            image = cv2.imread(currentDirectory+cfile, 0)
            ret, letter_inverse_threshold = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
            flat_letter_inverse_threshold = np.reshape(letter_inverse_threshold, -1)
            image_data.append(flat_letter_inverse_threshold)
            target_value.append(dname)
        
    return (np.array(image_data), np.array(target_value))

image_data, target_value = trainingDataFormation()
image_data = np.divide(image_data, 255)

svc_classifier = SVC(kernel = 'rbf', random_state=0)
svc_classifier.fit(image_data, target_value)

def crossValidation(model, folds, training_data, training_label):
    accuracy = cross_val_score(model, training_data, training_label, cv=int(folds))
    return accuracy

accuracy = crossValidation(svc_classifier, 5, image_data, target_value)

saveDir = '../Models/SVC/'
joblib.dump(svc_classifier, saveDir + 'svc2.pkl')