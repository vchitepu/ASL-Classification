# -*- coding: utf-8 -*-


##############################################################################
################################## EASY MODEL ################################
##############################################################################


import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


SCORE_DATADIR = 'TestingData'

CATEGORIES = ['A', 'F']

answers = ['a', 'f', 'a', 'a', 'a', 'a','a','f','a','a','f','f','f','f','a','a','a','f',]




##############################################################################
############################### PREPROCESSING ################################
##############################################################################


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




##############################################################################
################################ LOADING DATA ################################
##############################################################################


def load(img_size = 64, preprocess=True):
    data = [];
    dir = 'testDataforStudents/easyData'
    for x in range(166):
        path = os.path.join(dir, str(x) + '.jpg')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        # if preprocess:
        #     img = process_image(img)
        data.append(img)
    data = np.array(data)
    data = data.reshape(data.shape[0], img_size, img_size, 1)
    data = data/255

    return data


X = load()

def load_data(data = 'Testing', img_size = 100, preprocess = False):
    
    if data.upper() == 'TRAINING': d = os.path.join(os.getcwd(), TRAIN_DATADIR)
    elif data.upper() == 'TESTING': d = os.path.join(os.getcwd(), SCORE_DATADIR)
    else: print("Incorrect paramter")
    
    data = []
    labels = []
    for cat in CATEGORIES:
        path = os.path.join(d, cat)
        label = CATEGORIES.index(cat)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size,img_size))
                if preprocess:
                    img_array = process_image(img_array)
                data.append(img_array)
                labels.append(label)
            except Exception as e:
                pass
    
    data = np.array(data)
    data = data.reshape(data.shape[0], img_size, img_size, 1)
    data = data /255

    return np.array(data), np.array(labels)



##############################################################################
################################ LOADING NN ##################################
##############################################################################


model = load_model(os.path.join(os.path.join(os.getcwd(), 'saved_models'), 'easy_model_backup.h5'))
# acc = model.evaluate(X,y)
# print('Accuracy: ' + str(acc[1]))



##############################################################################
############################## WRITING TO FILE ###############################
##############################################################################


# Getting predicted classifications
preds = model.predict_classes(X)
preds = preds.reshape(-1)

# Converts class id numbers to letter
def conv_to_letter(ent):
    i = ent
    if i == 0:
        i = 'a'
    else:
        i = 'f'
    return i

predictions = np.array([conv_to_letter(x) for x in preds])
print('\nPredictions: \n' + str(predictions))
with open(os.path.join(os.path.join(os.getcwd(), 'outputs'), 'results_easy.txt'), 'w') as file:
    file.write("[")
    predictions.tofile(file, ",")
    file.write("]")












