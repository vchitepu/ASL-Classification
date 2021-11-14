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
from tensorflow.keras.utils import to_categorical



SCORE_DATADIR = './TestingData'

CATEGORIES = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']


##############################################################################
################################ LOADING DATA ################################
##############################################################################


def load_data(data = 'Testing', img_size = 100):
    
    if data.upper() == 'TRAINING': d = TRAIN_DATADIR
    elif data.upper() == 'TESTING': d = SCORE_DATADIR
    else: print("Incorrect paramter")
    
    data = []
    labels = []
    for cat in CATEGORIES:
        count = 0
        path = os.path.join(d, cat)
        label = CATEGORIES.index(cat)
        for img in os.listdir(path):
            if count == 1500 : break
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_array = cv2.resize(img_array, (img_size,img_size))
                data.append(img_array)
                labels.append(label)
                count+=1
            except Exception as e:
                pass

    # Normalizing data
    data = np.array(data)
    data = data.astype('float32')/255
    
    # One-Hot Encoding for Labels
    labels = np.array(labels)
    labels2 = labels
    labels = to_categorical(labels)
    
    
    
    return data, labels, labels2

X, y, labels = load_data(img_size = 64)


##############################################################################
################################ LOADING NN #################################
##############################################################################


model = load_model('./saved_models/hard_model_backup.h5')
acc = model.evaluate(X, y)


##############################################################################
############################## WRITING TO FILE ###############################
##############################################################################


# Getting predicted classifications
preds = model.predict_classes(X)
preds = preds.reshape(-1)

# Converts class id numbers to letter
def conv_to_letter(ent):
    return CATEGORIES[ent]

predictions = np.array([conv_to_letter(x) for x in preds])
print('\nPredictions: \n' + str(predictions))
print('Accuracy: ' + str(acc[1]))
with open('./outputs/results_hard.txt', 'w') as file:
    file.write(str(predictions))

