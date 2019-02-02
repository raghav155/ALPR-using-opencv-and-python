# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:21:50 2019

@author: Raghav
"""

import numpy as np
import cv2

DIRECTORY = './Dataset/'

for fname in range(27,63):
    currentDirectory = DIRECTORY + str(fname) + '/'
    for cfile in range(1,31):
         currentFile = currentDirectory + str(cfile) + '.png'
         image = cv2.imread(currentFile)
         resized_image = cv2.resize(image, (20,20))
         name = currentDirectory + str(cfile) + '_20by20.png'
         cv2.imwrite(name, resized_image)


