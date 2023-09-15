#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:11:43 2019

@author: wangzheng
"""
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_data():
    data = csv.reader(open('knowledge_data.csv', 'r'))
    data = [line[0].split('\t') for line in data]
    data = np.array(data)

    print(data.shape)
    
    x = data[:, :5]
    y = data[:, -1]
    
    x[0][0] = '0.08'
    
    x = x.astype(np.float)
    y = y.astype(np.int)
    
    return x, y

def SVM_phase1(x_train, x_test, y_train, y_test):    
    model = SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model.fit(x_train, y_train)
    model.predict(x_test)
    score1 = model.score(x_test, y_test)
    print("\nOriginal test score: ", score1)
    
    return score1

def KNN_phase1(x_train, x_test, y_train, y_test):    
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(x_train, y_train)
    model.predict(x_test)
    score1 = model.score(x_test, y_test)
    print("\nOriginal test score: ", score1)
    
    return score1

#x, y = load_data()
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#KNN_phase1(x_train, x_test, y_train, y_test)