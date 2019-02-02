# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:48:59 2019

@author: Raghav
"""
import os
import random

CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
              'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

FONTS = (os.listdir('fonts/'))

backgroundDimensions = os.popen('convert ' + 'background.jpg ' + 'ping -format "%w %h" info:').read()
backgroundDimensions = backgroundDimensions.split(' ')
backgroundWidth = backgroundDimensions[0]
backgroundHeight = backgroundDimensions[1]

backgroundOutfile = 'background_outfile.jpg'
command = "magick convert " + "background.jpg " + "-crop 20x20+0+0 " + backgroundOutfile
os.system(str(command))

for i in range(0,len(CHARACTERS)):
    char_output_dir = 'training_data/' + CHARACTERS[i] + '/'
    
    if(not os.path.exists(char_output_dir)):
        os.makedirs(char_output_dir)
        
        print('Generating Data ' + char_output_dir)
    
    
    for j in range(0,1000):
        
        font = 'fonts/' + random.choice(FONTS)
        
        # Get random blur amount
        blur = random.randint(0,3)
        
        # Add random shifts from the center
        x = str(random.randint(-1,1))
        y = str(random.randint(-1,1))
        
        command =  "magick convert " + str(backgroundOutfile) + " -fill "+str('white')+" -font "+ \
        str(font) + " -weight 900 -pointsize 20 "+"-gravity center" + " -blur 0x" + str(blur) \
        + " -annotate +" + x + "+" + y + " " + str(CHARACTERS[i]) + " " + char_output_dir + "output_file"+str(i)+str(j)+".jpg"
        
        os.popen(str(command))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        