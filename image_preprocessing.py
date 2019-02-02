# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:47:28 2019

@author: Raghav
"""

import numpy as np
import cv2

# Loading original image
original_image = cv2.imread('./car_data/5.jpg')
cv2.imshow('Original Image', original_image)

# Resizing the original image to improve time complexity
resized_image = cv2.resize(original_image, (500, 400))
cv2.imshow('Resized image', resized_image)
cropped_image = resized_image[110:350, 50:500]
cv2.imshow('cropped', cropped_image)

#resized_cropped_image = cv2.resize(cropped_image, (296,44))
#cv2.imshow('Resized cropped image', resized_cropped_image)

cv2.imwrite('output/ROI.jpg', cropped_image)

