# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:11:13 2019

@author: Raghav
"""

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
              'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

DIRECTORY = './Dataset/'

def trainingDataFormation():
    image_data = []
    target_value = []
    
    for fname in range(27,63):
        currentDirectory = DIRECTORY + str(fname) + '/'
        for cfile in range(1,31):
            currentFile = currentDirectory + str(cfile) + '_20by20.png'
            image = cv2.imread(currentFile, 0)
            ret, letter_inverse_threshold = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
            flat_letter_inverse_threshold = np.reshape(letter_inverse_threshold, -1)
            image_data.append(flat_letter_inverse_threshold)
            target_value.append(CHARACTERS[fname-1])
        
    return (np.array(image_data), np.array(target_value))


def crossValidation(model, folds, training_data, training_label):
    accuracy = cross_val_score(model, training_data, training_label, cv=int(folds))
    return accuracy


image_data, target_value = trainingDataFormation()
svc_classifier = SVC(kernel = 'linear', random_state=0)
svc_classifier.fit(image_data, target_value)
accuracy = crossValidation(svc_classifier, 5, image_data, target_value)

saveDir = './Models/SVC/'
joblib.dump(svc_classifier, saveDir + 'svc.pkl')