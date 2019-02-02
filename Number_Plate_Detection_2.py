# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:04:40 2019

@author: Raghav
"""
import numpy as np
import cv2

cropped_image = cv2.imread('./output/ROI.jpg')

gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

bilateral_filtered = cv2.bilateralFilter(gray, 11, 50, 50)
cv2.imshow('Bilateral Filtered', bilateral_filtered)

ret, plate_inverse_threshold = cv2.threshold(bilateral_filtered, 115, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold', plate_inverse_threshold)


# Vertical and Horizontal Scanning to reduce the region of interest
middle_x = int(plate_inverse_threshold.shape[0]/2)
middle_y = int(plate_inverse_threshold.shape[1]/2)


def getTopCoordinate():
    for x in range(middle_x,-1,-1):
        black_count = 0
        white_count = 0
        for y in range(0, plate_inverse_threshold.shape[1]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        ratio = 400
        
        if(black_count != 0):
            ratio = white_count/black_count
        
        #print((white_count,black_count, ratio))
        if(ratio > 10 or ratio < 0.3):
            return x
    
    return 0

def getBottomCoordinate():
    for x in range(middle_x,250,1):
        black_count = 0
        white_count = 0
        for y in range(0, plate_inverse_threshold.shape[1]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        ratio = 400
        
        if(black_count != 0):
            ratio = white_count/black_count
        
        print(ratio)
        if(ratio > 10 or ratio < 0.3):
            return x
        
    
    return 150

def getLeftCoordinate():
    for y in range(middle_y,-1,-1):
        black_count = 0
        white_count = 0
        for x in range(0, plate_inverse_threshold.shape[0]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        ratio = 150
        if(black_count != 0):
            ratio = white_count/black_count
        
        if(ratio > 30):
            return y
    return 0

def getRightCoordinate():
    for y in range(middle_y,450,1):
        black_count = 0
        white_count = 0
        for x in range(0, plate_inverse_threshold.shape[0]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        ratio = 150
        if(black_count != 0):
            ratio = white_count/black_count
        
        if(ratio > 40):
            return y
    return plate_inverse_threshold.shape[1]


top = getTopCoordinate()
bottom = getBottomCoordinate()
left = getLeftCoordinate()
right = getRightCoordinate()
#img1 = cropped_image[top:bottom, left:right]
#cv2.imshow('cropped', img1)

for l in range(1,7):
    if(top-l >= 0):
        top = top-l
    
    if(bottom+l < cropped_image.shape[0]):
        bottom = bottom+l
    
    if(left-l >= 0):
        left = left-l
    
    if(right+l < cropped_image.shape[1]):
        right = right+l

img = cropped_image[top:bottom, left:right]
cv2.imshow('crop', img)

cv2.imwrite('./output/plate_ROI.jpg', img)

