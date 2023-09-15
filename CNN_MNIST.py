#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Dec 1 16:10:24 2019

@author: Wang, Zheng (zw1454@nyu.edu)
"""
import random
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

uu_label = [2,3,4,5,6,7,8,9]
uu_label_test = 2

input_shape = (28, 28, 1)


'''------------------ make data ------------------------ 
In this part, we generate our data and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

def make_data():
    (old_x_train, old_y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    old_x_train = old_x_train.reshape(old_x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Making sure that the values are float so that we can get decimal points after division
    old_x_train = old_x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalizing the RGB codes by dividing it to the max RGB value.
    old_x_train /= 255
    x_test /= 255
    
    print('x_train shape:', old_x_train.shape)
    print('Number of images in x_train: ', old_x_train.shape[0])
    print('Number of images in x_test: ', x_test.shape[0])
        
    for i in range(y_test.shape[0]):
        if y_test[i] in uu_label:
            y_test[i] = uu_label_test
    
    x_train = []
    y_train = []
    uu_x = []
    uu_y = []
    for i in range(old_x_train.shape[0]):
        if old_y_train[i] not in uu_label:
            x_train.append(old_x_train[i])
            y_train.append(old_y_train[i])
        else:
            uu_x.append(old_x_train[i])
            uu_y.append(old_y_train[i])
                   
    x_train = np.array(x_train)     
    y_train = np.array(y_train)
    uu_x = np.array(uu_x)
    uu_y = np.array(uu_y)
    
    print("Shape of all data: ", x_train.shape[0]+uu_x.shape[0])
    print("Shape of unknown unknown data: ", uu_x.shape[0])
    
    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x_train, x_test, y_train, y_test

'''------------------- phase 1 -------------------------
In this phase, we will train the neural network on the 6 known classes
and test its performance on the test classes.
'''

def CNN_phase1(x_train, x_test, y_train, y_test):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(rate=0.2))
    model.add(Dense(uu_label_test+1, activation=tf.nn.softmax))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=4)
    
    score1 = model.evaluate(x_test, y_test)[1]
    print("\nOriginal accuracy score: ", score1)
    
    origin_pred = model.predict_classes(x_test)
    f_measure = f1_score(y_test, origin_pred, average='macro')
    print("\nOriginal F-measure: ", f_measure)
    
    return (score1, f_measure)
    
'''------------------- phase 2 ------------------------
In this phase, we will: first, random sample from the test set and add it as a new
class to the existing training set; second, perform a cross validation on the training
set and sort out all the data assigned to the new sample class; third, retrain the classifer
on the new training set and evaluate its performance on the test set.
'''

def random_sampling(n, x_train, x_test, y_train, y_test):      
    # n: number of samples from the test set
    A = [list(i) for i in x_test]
    a = list(y_test) 
    newA, newa = zip(*random.sample(list(zip(A, a)), n))
    newA_l, newa_l = list(newA), list(newa)
    sample, sample_label = np.array(newA_l), np.array(newa_l) # obtain a sample of test set
    
#    new_x_test = []
#    new_y_test = []
#    for i in range(len(A)):
#        print(i)
#        if not any(np.array_equal(x, A[i]) for x in sample):
#            new_x_test.append(A[i])
#            new_y_test.append(a[i]) # reconstruct the test set by removing the samples
#            
#    new_x_test = np.array(new_x_test)
#    new_y_test = np.array(new_y_test)

    print("\nRandom sampling complete!")
    return sample, sample_label, x_test, y_test

def cross_validation(X, Y, k=3):
    # divide the training data into k-folds
    kf = KFold(n_splits=k)
    unknown = []
    unknown_label = []
    uu_count = 0
    
    for train_index, test_index in kf.split(X, Y):
        xc_train, xc_test = X[train_index], X[test_index]
        yc_train, yc_test = Y[train_index], Y[test_index]
        
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(rate=0.2))
        model.add(Dense(uu_label_test+1, activation=tf.nn.softmax))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=xc_train, y=yc_train, epochs=4)

        # sort out the samples classified into the new class
        for x in xc_test:
            l = model.predict_classes(np.array([x]))
            if l == np.array([uu_label_test]):                                    ####
                uu_count += 1
                unknown.append(list(x))
                unknown_label.append(uu_label_test)                               #####
    
#    print("\nThe number of elements in the unknown unknown class: ", uu_count)
    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
#    print(unknown.shape)
    
    return unknown, unknown_label

    
def CNN_phase2(sample_size, x_train, x_test, y_train, y_test):
    sample_num = int(sample_size * y_test.shape[0])
    
    sample, sample_label, new_x_test, new_y_test \
            = random_sampling(sample_num, x_train, x_test, y_train, y_test)
            
    sample_label = [uu_label_test for i in range(len(sample_label))]              #####
    sample_label = np.array(sample_label)
    
    classes = np.concatenate((x_train, sample), axis=0) # construct new training classes
    classes_label = np.append(y_train, sample_label)
    
    # randomly shuffle the classes
    zipped = list(zip(classes, classes_label))  
    random.shuffle(zipped)
    classes, classes_label = zip(*zipped)
    classes = np.array(classes)
    classes_label = np.array(classes_label)
    
    un_class, un_label = cross_validation(classes, classes_label)
    
    # construct the final training set by adding the unknown unknowns
    x_train2 = np.concatenate((x_train, un_class), axis=0)
    y_train2 = np.append(y_train, un_label)    
    
    # train & test on the new sets
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(rate=0.3))
    model.add(Dense(uu_label_test+1, activation=tf.nn.softmax))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train2, y=y_train2, epochs=3)
    
    score2 = model.evaluate(new_x_test, new_y_test)[1]
    print("\nFinal accuracy score: ", score2)
    
    origin_pred = model.predict_classes(new_x_test)
    print(origin_pred.shape)
    f_measure = f1_score(new_y_test, origin_pred, average='macro')
    print("\nFinal F-measure: ", f_measure)
    
    return score2, f_measure
  
def CNN_main(sample_size):
    x_train, x_test, y_train, y_test = make_data()
    CNN_phase1(x_train, x_test, y_train, y_test)
    score2, f_measure = CNN_phase2(sample_size, x_train, x_test, y_train, y_test)
    return score2, f_measure

 
####################################
CNN_main(0.3) 


