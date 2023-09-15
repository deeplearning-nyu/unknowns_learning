#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday May 3 23:00:12 2019

@author: Wang, Zheng (zw1454@nyu.edu)
"""
import random
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.manifold import MDS
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris

uu_label = 2

'''------------------ make data ------------------------ 
In this part, we generate our data (10 classes + 1 unknown unknown class) and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

def make_data():
    iris = load_iris()

    x = iris.data
    y = iris.target
    
    old_x_train, x_test, old_y_train, y_test = train_test_split(x, y, test_size=0.3)

    # remove class 0 from training set
    x_train = []
    y_train = []
    uu_x = []
    uu_y = []
    for i in range(old_x_train.shape[0]):
        if old_y_train[i] != uu_label:                                        ###
            x_train.append(old_x_train[i])
            y_train.append(old_y_train[i])
        else:
            uu_x.append(old_x_train[i])
            uu_y.append(old_y_train[i])
       
    x_train = np.array(x_train)     
    y_train = np.array(y_train)
    uu_x = np.array(uu_x)
    uu_y = np.array(uu_y)
    
#    print("Shape of all data: ", x.shape)
#    print("Shape of unknown unknown data: ", uu_x.shape)
    
    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
#    print("Shape of the original training data: ", x_train.shape)
#    print("Shape of the original test data: ", x_test.shape)
    
    '''----------------------------------------------------'''    
#    print("Performing Multidimensional Scaling...")
#    embedding = MDS(n_components=2)
#    x_transformed = embedding.fit_transform(x)
#    
#    pyplot.figure(1)
#    pyplot.scatter(x_transformed[:,0], x_transformed[:,1],c=y)
#    pyplot.title("The structure of the iris classes")
#    pyplot.show()
    '''----------------------------------------------------'''
    
    return x_train, x_test, y_train, y_test

'''------------------- phase 1 -------------------------
In this phase, we will train the classifier (here we pick SVM) on the 10 known classes
and test its performance on the 10+1 test classes.
This phase serves as the control group of our experiment.
'''

def SVM_phase1(x_train, x_test, y_train, y_test):    
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model.fit(x_train, y_train)
    model.predict(x_test)
    score1 = model.score(x_test, y_test)
    print("\nOriginal test score: ", score1)
    
    return score1
    
    
    
'''------------------- phase 2 ------------------------
In this phase, we will: first, random sample from the test set and add it as a new
class to the existing training set; second, perform a cross validation on the training
set and sort out all the data assigned to the new sample class; third, retrain the classifer
on the new training set and evaluate its performance on the test set.
This phase is the experiment group.
'''

def random_sampling(n, x_train, x_test, y_train, y_test):      
    # n: number of samples from the test set
    A = [list(i) for i in x_test]
    a = list(y_test) 
    newA, newa = zip(*random.sample(list(zip(A, a)), n))
    newA_l, newa_l = list(newA), list(newa)
    sample, sample_label = np.array(newA_l), np.array(newa_l) # obtain a sample of test set
    
    new_x_test = []
    new_y_test = []
    for i in range(len(A)):
        if not any(np.array_equal(x, A[i]) for x in sample):
            new_x_test.append(A[i])
            new_y_test.append(a[i]) # reconstruct the test set by removing the samples
            
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)
#    print("\nShape of sample class: ", sample.shape)
#    print("Shape of new test set: ", new_x_test.shape)
    
    return sample, sample_label, new_x_test, new_y_test

def cross_validation(X, Y, k=3):
    # divide the training data into k-folds
    kf = KFold(n_splits=k)
    unknown = []
    unknown_label = []
    uu_count = 0
    
    for train_index, test_index in kf.split(X, Y):
        xc_train, xc_test = X[train_index], X[test_index]
#        print("\nShape of cross validation training class: ", xc_train.shape)
#        print("\nShape of cross validation test class: ", xc_test.shape)
        yc_train, yc_test = Y[train_index], Y[test_index]
        
        cross_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
             decision_function_shape='ovr', degree=3, gamma='auto',
             kernel='rbf', max_iter=-1, probability=True, random_state=None,
             shrinking=True, tol=0.001, verbose=False)
        cross_model.fit(xc_train, yc_train)
        
        # sort out the samples classified into the new class
        for x in xc_test:
            l = cross_model.predict(np.array([x]))
            if l == np.array([uu_label]):                                    ####
                uu_count += 1
                unknown.append(list(x))
                unknown_label.append(uu_label)                               #####
    
#    print("\nThe number of elements in the unknown unknown class: ", uu_count)
    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
#    print(unknown.shape)
    
    return unknown, unknown_label

    
def SVM_phase2(sample_size, x_train, x_test, y_train, y_test):
    sample, sample_label, new_x_test, new_y_test \
            = random_sampling(sample_size, x_train, x_test, y_train, y_test)
            
    sample_label = [uu_label for i in range(len(sample_label))]              #####
    sample_label = np.array(sample_label)
    
    classes = np.concatenate((x_train, sample), axis=0) # construct new training classes
    classes_label = np.append(y_train, sample_label)
#    print("Shape of new training classes: ", classes.shape)
    
    # randomly shuffle the classes
    zipped = list(zip(classes, classes_label))  
    random.shuffle(zipped)
    classes, classes_label = zip(*zipped)
    classes = np.array(classes)
    classes_label = np.array(classes_label)
    
    '''----------------------------------------------------'''
#    embedding = MDS(n_components=2)
#    c_transformed = embedding.fit_transform(classes)
#    
#    pyplot.figure(2)
#    pyplot.scatter(c_transformed[:,0],c_transformed[:,1],c=classes_label)
#    pyplot.title("The structure of the new training classes")
#    pyplot.show()  # Note that the color of the sample class is yellow
    '''----------------------------------------------------'''
    
    un_class, un_label = cross_validation(classes, classes_label, k=3)
    
    # construct the final training set by adding the unknown unknowns
    x_train2 = np.concatenate((x_train, un_class), axis=0)
    y_train2 = np.append(y_train, un_label)
#    print("\nShape of final training classes: ", x_train2.shape)
#    print("Shape of final test data: ", new_x_test.shape)
    
    '''----------------------------------------------------'''
#    embedding2 = MDS(n_components=2)
#    transformed2 = embedding2.fit_transform(x_train2)
#    
#    pyplot.figure(4)
#    pyplot.scatter(transformed2[:,0],transformed2[:,1],c=y_train2)
#    pyplot.title("The structure of the final training data")
#    pyplot.show()
    '''----------------------------------------------------'''
    
    
    # train & test on the new sets
    model2 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model2.fit(x_train2, y_train2)
    model2.predict(new_x_test)
    score2 = model2.score(new_x_test, new_y_test)
    print("\nSVM Final test score: ", score2)
    
    return score2
  
def SVM_main(sample_size):
    x_train, x_test, y_train, y_test = make_data()
    SVM_phase1(x_train, x_test, y_train, y_test)
    score2 = SVM_phase2(sample_size, x_train, x_test, y_train, y_test)
    return score2

 
####################################
def sample_test_comparison():
    import KNN_iris
    
    sample_size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    SVM_score = []
    KNN_score = []
    
    x_train, x_test, y_train, y_test = make_data()
    score1 = SVM_phase1(x_train, x_test, y_train, y_test)
    for s in sample_size:
        n = y_train.shape[0]
        p1 = 0
        p2 = 0
        count1 = 0
        count2 = 0
        
        for i in range(5):
            try:
                p1 += SVM_phase2(int(s*n), x_train, x_test, y_train, y_test)
                count1 += 1
            except:
                p1 += 0

            try:
                p2 += KNN_iris.KNN_phase2(int(s*n), x_train, x_test, y_train, y_test)
                count2 += 1
            except:
                p2 += 0
                
        print('SVM', p1, count1)
        SVM_score.append(p1/count1)
        KNN_score.append(p2/count2)

    pyplot.figure(1)
    pyplot.plot(sample_size, SVM_score)
    pyplot.plot(sample_size, KNN_score)
    pyplot.hlines(score1, 0.1, 0.5)
    pyplot.xlabel('Sample size (%)')
    pyplot.ylabel('Precision score')
    pyplot.legend(['SVM', 'KNN', 'Benchmark'])
    pyplot.show()

sample_test_comparison()


 

