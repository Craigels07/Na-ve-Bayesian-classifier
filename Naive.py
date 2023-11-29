# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:37:15 2022

@author: craig
"""

from scipy.stats import mode
import numpy as np

from time import time
import pandas as pd
import os
import matplotlib.pyplot as matplot
import matplotlib


import random

from IPython.display import display, HTML
from itertools import chain
from sklearn.metrics import confusion_matrix
import seaborn as sb
from tensorflow.keras.datasets import mnist
import numpy as np



(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))

train_norm = trainX.astype('float32')
train_norm = train_norm / 255.0


def naivebayes(train, train_lb, test, test_lb, smoothing):
        n_class = np.unique(train_lb)
        tr = train
        te = test
        tr_lb = train_lb
        te_lb = test_lb
        smoothing = smoothing
        st = time()
        m, s, prior, count = [], [], [], []
        for i, val in enumerate(n_class):
            sep = [tr_lb == val] 
            count.append(len(tr_lb[sep]))
            prior.append(len(tr_lb[sep]) / len(tr_lb))
            m.append(np.mean(tr[sep], axis=0))
            s.append(np.std(tr[sep], axis=0))
        
        pred = []
        likelihood = []
        #prtab = []
        lcs = []
        for n in range(len(te_lb)):
            classifier = []
            sample = te[n] #test sample
            ll = []
            for i, val in enumerate(n_class):
                m1 = m[i]
                var = np.square(s[i]) + smoothing
                prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(sample - m1)/(2 * var))
                #prtab.append(prob)
                result = np.sum(np.log(prob))
                classifier.append(result)
                ll.append(prob)

            pred.append(np.argmax(classifier))
            likelihood.append(ll)
            lcs.append(classifier)
        
        return pred, likelihood
    
    
def error_rate(confusion_matrix):
    a = confusion_matrix
    b = a.sum(axis=1)
    df = []
    for i in range(0,10):
        temp = 1-a[i][i]/b[i]
        df.append(temp)
    
    df = pd.DataFrame(df)
    df.columns = ['% Error rate']
    return df*100

nb = naivebayes(train=trainX, train_lb=trainY, test=testX, test_lb=testY, smoothing=1000)
nb_pred = nb[0]
cm = confusion_matrix(testY, nb_pred)
print("Test Accuracy:", round((sum(np.diagonal(cm)) / len(nb_pred)) * 100, 4), '%') 


#cm # X-axis Predicted vs Y-axis Actual Values
matplot.subplots(figsize=(10, 6))
sb.heatmap(cm, annot = True, fmt = 'g')
matplot.xlabel("Predicted")
matplot.ylabel("Actual")
matplot.title("Confusion Matrix")
matplot.show()   
error_rate(cm)