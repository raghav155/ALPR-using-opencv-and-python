# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:13:45 2019

@author: Raghav
"""
import numpy as np
import cv2

def filter_contours(contour):   
    area = cv2.contourArea(contour)
    
    return area >= 2000 and area <= 50000

# Loading the bilateral filtered image
gray_scale = cv2.imread('./output/ROI.jpg',0)
cv2.imshow('gray', gray_scale)

bilateral_filtered = cv2.bilateralFilter(gray_scale, 11, 50, 50)
cv2.imshow("Bilateral Filter", bilateral_filtered)

# Edge detection using canny algorithm
canny = cv2.Canny(bilateral_filtered, 200, 250)
cv2.imshow('Canny', canny)

# finding contours
img, contours, heirarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

copy = bilateral_filtered.copy()
for c in contours:
    print(cv2.contourArea(c))
    cv2.drawContours(copy, [c], -1, (255,0,0), 1)
    cv2.imshow('Contours', copy)
    #cv2.waitKey(0)


# sorting contours based on Area
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
sorted_contours_filtered = filter(filter_contours, sorted_contours)
sorted_contours_filtered = list(sorted_contours_filtered)

copy2 = bilateral_filtered.copy()
for c in sorted_contours_filtered:
    print(cv2.contourArea(c))
    cv2.drawContours(copy2, [c], -1, (255,0,0), 1)
    cv2.waitKey(0)
    cv2.imshow('Contours by area', copy2)


NumberPlateContour = None

copy3 = bilateral_filtered.copy()
for c in sorted_contours_filtered:
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    print(approx)
    
    if(len(approx) <= 4): # Finding contour with 4 corners
      NumberPlateContour = cv2.boundingRect(c)
      break

x, y, width, height = NumberPlateContour
roi = bilateral_filtered[y:y+height, x:x+width]

cv2.imshow('ROI', roi)

image_scaled = cv2.resize(roi, (300, 50))
cv2.imshow('ROI', image_scaled)

cv2.imwrite('output/ROI.jpg', image_scaled)




















