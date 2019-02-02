# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:00:26 2019

@author: Raghav
"""

import numpy as np
import cv2
import glob, os
import queue

plate_original = cv2.imread('./output/plate_ROI.jpg')
gray = cv2.cvtColor(plate_original, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

bilateral_filtered = cv2.bilateralFilter(gray, 11, 50, 50)
cv2.imshow('Bilateral Filtered', bilateral_filtered)

ret, plate_inverse_threshold = cv2.threshold(bilateral_filtered, 115, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold', plate_inverse_threshold)


# Again refine the pictures to remove borders
x_middle = int(plate_inverse_threshold.shape[0]/2)
y_middle = int(plate_inverse_threshold.shape[1]/2)

def getTopCoordinate():
    for x in range(x_middle, -1, -1):
        white_count = 0
        black_count = 0
        for y in range(0, plate_inverse_threshold.shape[1]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        if(black_count == 0):
            #print(plate_inverse_threshold.shape[1])
            #continue
            return x
        
        ratio = white_count / black_count
        #print(ratio)
        if(ratio >= 15):
            return x
    
    return 0


def getBottomCoordinate():
    for x in range(x_middle, plate_inverse_threshold.shape[0]):
        white_count = 0
        black_count = 0
        for y in range(0, plate_inverse_threshold.shape[1]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        if(black_count == 0):
            #print(plate_inverse_threshold.shape[1])
            #continue
            return x
        
        ratio = white_count / black_count
        #print(ratio)
        if(ratio >= 15):
            return x
    
    return plate_inverse_threshold.shape[0]

def getLeftCoordinate():
    for y in range(y_middle, -1, -1):
        white_count = 0
        black_count = 0
        for x in range(0, plate_inverse_threshold.shape[0]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        if(black_count == 0):
            #print(plate_inverse_threshold.shape[0])
            #continue
            return y
        
        ratio = white_count / black_count
        if(ratio >= 15):
            return y
    
    return 0

def getRightCoordinate():
    for y in range(y_middle, plate_inverse_threshold.shape[1]):
        white_count = 0
        black_count = 0
        for x in range(0, plate_inverse_threshold.shape[0]):
            if(plate_inverse_threshold[x][y] == 255):
                white_count += 1
            else:
                black_count += 1
        
        if(black_count == 0):
            return y
        
        ratio = white_count / black_count
        if(ratio >= 15):
            return y
    
    return plate_inverse_threshold.shape[1]



top = getTopCoordinate()
bottom = getBottomCoordinate()
left = getLeftCoordinate()
right = getRightCoordinate()

plate_inverse_threshold = plate_inverse_threshold[top:bottom, left:right]
cv2.imshow('Refined Image', plate_inverse_threshold)

# More refining
x_middle = int(plate_inverse_threshold.shape[0]/2)
y_middle = int(plate_inverse_threshold.shape[1]/2)

margin_x = int(0.4*x_middle)
margin_y = int(0.4*y_middle)

def removeWhiteBorders(x_start,x_end,x_step,y_start,y_end,y_step,point,reverse):
    for i1 in range(x_start,x_end,x_step):
        black_cell_count = 0
        for i2 in range(y_start,y_end,y_step):
            if(reverse == False and plate_inverse_threshold[i1][i2] == 0):
                black_cell_count += 1
            elif(reverse == True and plate_inverse_threshold[i2][i1] == 0):
                black_cell_count += 1
                
        if(black_cell_count == 0):
            point = i1
    
    return point

top = removeWhiteBorders(0,x_middle-margin_x,1,0,plate_inverse_threshold.shape[1],1,0,False)
bottom = removeWhiteBorders(plate_inverse_threshold.shape[0]-1,x_middle+margin_x,-1,0,plate_inverse_threshold.shape[1],1,plate_inverse_threshold.shape[0],False)
left = removeWhiteBorders(0,y_middle-margin_y,1,0,plate_inverse_threshold.shape[0],1,0,True)
right = removeWhiteBorders(plate_inverse_threshold.shape[1]-1,y_middle+margin_y,-1,0,plate_inverse_threshold.shape[0],1,plate_inverse_threshold.shape[1],True)  

plate_inverse_threshold = plate_inverse_threshold[top:bottom, left:right]
cv2.imshow('Refined Image 2', plate_inverse_threshold)


# Lets do some blob detecton via Flood Fill algorithm and also eliminating noise regions
isdone = set()
rows, cols = plate_inverse_threshold.shape
list_blobs = []

'''
def floodFill(x, y, coordList):
    num = x*cols + y
    if(plate_inverse_threshold[x][y] == 0 or (num in isdone)):
        return
    
    isdone.add(num)
    coordList.append((x, y))
    
    if(x-1 >= 0):
        floodFill(x-1, y, coordList)
    if(y+1 < cols):
        floodFill(x, y+1, coordList)
    if(x+1 < rows):
        floodFill(x+1, y, coordList)
    if(y-1 >= 0):
        floodFill(x, y-1, coordList)
    
    return



for y in range(0, cols):
    for x in range(0, rows):
        unique_num = x*cols + y
        if(unique_num not in isdone):
            temp_list = []
            floodFill(x, y, temp_list)
            if(len(temp_list) > 0):
               list_blobs.append(temp_list)
'''              

# Blob detection usin BFS to overcome stackoverflow problem in flood fill algorithm( recursive DFS)
def getUniqueNum(x, y):
    return x*cols + y

def BFS(x, y, coordList):
    que = queue.Queue()
    que.put((x,y))
    
    while(not que.empty()):
        currentCoord = que.get()
        coordList.append(currentCoord)
        
        top = (currentCoord[0]-1,currentCoord[1])
        unique_top = getUniqueNum(top[0], top[1])
        
        right = (currentCoord[0], currentCoord[1]+1)
        unique_right = getUniqueNum(right[0], right[1])
        
        bottom = (currentCoord[0]+1, currentCoord[1])
        unique_bottom = getUniqueNum(bottom[0], bottom[1])
        
        left = (currentCoord[0], currentCoord[1]-1)
        unique_left = getUniqueNum(left[0], left[1])
        
        if(top[0] >= 0 and (unique_top not in isdone)):
            if(plate_inverse_threshold[top[0]][top[1]] != 0):
                isdone.add(unique_top)
                que.put(top)
        
        if(right[1] < cols and (unique_right not in isdone)):
            if(plate_inverse_threshold[right[0]][right[1]] != 0):
                isdone.add(unique_right)
                que.put(right)
        
        if(bottom[0] < rows and (unique_bottom not in isdone)):
            if(plate_inverse_threshold[bottom[0]][bottom[1]] != 0):
                isdone.add(unique_bottom)
                que.put(bottom)
        
        if(left[1] >= 0 and (unique_left not in isdone)):
            if(plate_inverse_threshold[left[0]][left[1]] != 0):
                isdone.add(unique_left)
                que.put(left)



for y in range(0, cols):
    for x in range(0, rows):
        unique_num = getUniqueNum(x, y)
        if(unique_num not in isdone):
            isdone.add(unique_num)
            if(plate_inverse_threshold[x][y] != 0):
                temp_list = []
                BFS(x, y, temp_list)
                list_blobs.append(temp_list)

     
# extracting bounding rectangle coordinates of each character
def getLeftMostCoordinate(tup_list):
    leftpt = cols
    for tup in tup_list:
        if(leftpt > tup[1]):
            leftpt = tup[1]
    
    return leftpt

def getTopMostCoordinate(tup_list):
    toppt = rows
    for tup in tup_list:
        if(toppt > tup[0]):
            toppt = tup[0]
    
    return toppt

def getRightMostCoordinate(tup_list):
    rightpt = -1
    for tup in tup_list:
        if(rightpt < tup[1]):
            rightpt = tup[1]
    
    return rightpt

def getBottomMostCoordinate(tup_list):
    bottompt = -1
    for tup in tup_list:
        if(bottompt < tup[0]):
            bottompt = tup[0]
    
    return bottompt

def getHeightWidth(tup_list):
    #print((tup_list[1][0], tup_list[0][0]),(tup_list[1][1], tup_list[0][1] ))
    return (tup_list[1][0] - tup_list[0][0],tup_list[1][1] - tup_list[0][1] )

def filter_rectangle(tup_list):
    HW = getHeightWidth(tup_list)
    diff = HW[0] - HW[1]
    area = (tup_list[1][0] - tup_list[0][0]) * (tup_list[1][1] - tup_list[0][1])
    #return diff > 0 and HW[0] >= int(plate_inverse_threshold.shape[0]/4.5) and HW[1] >= int(plate_inverse_threshold.shape[1]/40)
    return diff > 0 and area > 10

bounding_rectangle_coordinates = []


for lst in list_blobs:
    leftmost_coord = getLeftMostCoordinate(lst)
    topmost_coord = getTopMostCoordinate(lst)
    rightmost_coord = getRightMostCoordinate(lst)
    bottommost_coord = getBottomMostCoordinate(lst)
    
    left_top = (topmost_coord, leftmost_coord)
    right_bottom = (bottommost_coord, rightmost_coord)
    coordinate = (left_top, right_bottom)
    bounding_rectangle_coordinates.append(coordinate)


#mean = mean_list(bounding_rectangle_coordinates)
filtered_bounding_rectangle_coordinates = filter(filter_rectangle, bounding_rectangle_coordinates)
filtered_bounding_rectangle_coordinates = list(filtered_bounding_rectangle_coordinates)

# segmentation using rectangle coordinates
copy = plate_inverse_threshold.copy()
for tup in filtered_bounding_rectangle_coordinates:
    print((tup[1][0]-tup[0][0])*(tup[1][1]-tup[0][1]))
    cv2.rectangle(copy, (tup[0][1],tup[0][0]), (tup[1][1],tup[1][0]), (255, 0, 0), 1)
    cv2.imshow('Bounding Rectangles', copy)
    cv2.waitKey(0)

# devise an algorithm to get the similar elements only




# Clear all the files
test = './output/segments/*'
r = glob.glob(test)
for i in r:
   os.remove(i)


def fillBorders(segment):
    
    
    for x in range(0, 2):
        for y in range(0, segment.shape[1]):
            segment[x][y] = 0
    
    for x in range(segment.shape[0]-1, segment.shape[0]-3, -1):
        for y in range(0, segment.shape[1]):
            segment[x][y] = 0
    
    for y in range(0, 2):
        for x in range(0, segment.shape[0]):
            segment[x][y] = 0
    
    for y in range(segment.shape[1]-1, segment.shape[1]-3, -1):
        for x in range(0, segment.shape[0]):
            segment[x][y] = 0

for index, tup in enumerate(filtered_bounding_rectangle_coordinates,0):
    y = tup[0][0]
    x = tup[0][1]
    
    height = tup[1][0] - y
    width = tup[1][1] - x
    
    if(y-2 >= 0 and y+height+2 < copy.shape[0] and x - 2 >= 0 and x+width+2 < copy.shape[1]):
        segment = plate_inverse_threshold[y-2:y+height+3, x-2:x+width+3]
        fillBorders(segment)
        segment = cv2.resize(segment, (20,20))
        #cv2.imshow('Segment ' + str(index), segment)
        #cv2.waitKey(0)
        cv2.imwrite('output/segments/'+str(index)+'.png',segment)










