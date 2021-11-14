# -*- coding: utf-8 -*-
"""
Preprocessing

Created on Thu Mar 14 14:51:23 2019

@author: Alex Isaly
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

imagedir = "4_Brogrammers"

def process_image(img):
    #Gaussian blur
    blur = cv2.GaussianBlur(img, (11,11), cv2.BORDER_DEFAULT)
    
    #Threshhold + Dilate + Erode
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    #Apply background to make background white
    imgRGB= cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask = sure_bg
    imgCopy = img.copy()
    imgCopy[mask==0] = 255
    
    return imgCopy