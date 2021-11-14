# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:16:17 2019

@author: 19548
"""

def process_image(img):
    
    def auto_canny(image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
    
    edges = auto_canny(img)
    out = np.bitwise_or(img, canny[:,:,np.newaxis])
    
    return out

process_image