#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
from scipy.stats import chi2
import tensorflow as tf
import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
import efficientnet.keras as enet
import warnings
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import math
warnings.filterwarnings("ignore")


# calculating mahalanobis distance b/w a sample point and a distribution
def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal



#Between class scatter matrix b/w unknown and known class
def S_B(known_mean_vec, d):
    k = d*3
    
    mean_overall = -1 * np.ones(k) # mean_overall of unknown class
    S_B = np.zeros((k,k))
    for i in range(known_mean_vec.shape[0]):
            mean_vec = known_mean_vec[i].reshape(k,1) 
            mean_overall = mean_overall.reshape(k,1)
            S_B += 0.1 * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
        #trace += np.trace(S_B)
        
        #print(known_mean_vec[i].shape)
        
    return S_B 



def make_data(known_mean, known_cov, d):
    #set constant covariance and size for unknown class
    k = d*3
    uu_cov = 2*np.identity(k)
    uu_mean = -1 * np.ones(k)
    
    
    
    
    mean0 = known_mean[0]
    mean1 = known_mean[1]
    mean2 = known_mean[2]
    mean3 = known_mean[3]
    mean4 = known_mean[4]
    mean5 = known_mean[5]
    mean6 = known_mean[6]
    mean7 = known_mean[7]
    mean8 = known_mean[8]
    mean9 = known_mean[9]

    known_mean_vec =  np.asarray(known_mean)
    known_cov_vec =  np.asarray(known_cov)

    
    SB = S_B(known_mean_vec, d)
    S_b = np.trace(SB)
   
    print('\nS_B separability b/w known classes and u.u class: ', S_b)
    ################
    
    # construct the 10 known classes
    class_size = 100
    class0 = multivariate_normal(mean0, known_cov[0], class_size)
    class1 = multivariate_normal(mean1, known_cov[1], class_size)
    class2 = multivariate_normal(mean2, known_cov[2], class_size)
    class3 = multivariate_normal(mean3, known_cov[3], class_size)
    class4 = multivariate_normal(mean4, known_cov[4], class_size)
    class5 = multivariate_normal(mean5, known_cov[5], class_size)
    class6 = multivariate_normal(mean6, known_cov[6], class_size)
    class7 = multivariate_normal(mean7, known_cov[7], class_size)
    class8 = multivariate_normal(mean8, known_cov[8], class_size)
    class9 = multivariate_normal(mean9, known_cov[9], class_size)
    
    known_x = np.concatenate([class0, class1, class2, class3, class4, class5, class6,
                       class7, class8, class9])
    known_y = []
    for i in range(class_size * 10):
        known_y.append(i // class_size)
    known_y = np.array(known_y)
   
    
    
    x_train, x_test, y_train, y_test = train_test_split(known_x, known_y, test_size=0.3)
    
    #construct the unknown class
    uu_x = multivariate_normal(uu_mean, uu_cov, 150)
    uu_y = np.array([10 for i in range(len(uu_x))])  
    
    all_x, all_y = np.concatenate((known_x, uu_x), axis=0), np.concatenate((known_y, uu_y), axis=0)
    x_test = np.concatenate((x_test, uu_x))
    y_test = np.concatenate((y_test, uu_y))
    
    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print("Shape of the original training data: ", x_train.shape)
    print("Shape of the original test data: ", x_test.shape)
    

    return x_train, x_test, y_train, y_test, S_b


def SVM_phase1(x_train, x_test, y_train, y_test): 
    print(x_train.shape[1:])
    
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model.fit(x_train, y_train)
    origin_pred = model.predict(x_test)
    score1 = model.score(x_test, y_test)
    print("\nOriginal accuracy score: ", score1)
    
    f_measure = f1_score(y_test, origin_pred, average='macro')
    print("\nOriginal F-measure: ", f_measure)
    
    return f_measure


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
        if A[i] not in sample:
            new_x_test.append(A[i])
            new_y_test.append(a[i]) # reconstruct the test set by removing the samples
            
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)
    
    return sample, sample_label, new_x_test, new_y_test


def cross_validation(X, Y, k=3):
    # divide the training data into k-folds
    kf = KFold(n_splits=k)
    unknown = []
    unknown_label = []
    uu_count = 0
    
    for train_index, test_index in kf.split(X, Y):
        xc_train, xc_test = X[train_index], X[test_index]
        yc_train, yc_test = Y[train_index], Y[test_index]
        
############### code block to use Mahalanobis Distance instead of cross-validation ###################################        
        
        threshold = chi2.ppf((1-0.01), df=xc_test.shape[1]-1)  #threshold for distinguishing b/w dummy class and known class
        print('Threshold:', threshold)
        
        # sort out the samples classified into the new class
        
        for x in xc_test:
            w = mahalanobis(x=x, data=X[Y != 10])
             
            if w >= threshold:
                   
                    uu_count += 1
                    unknown.append(list(x))
                    unknown_label.append(10)
                    
####### remove comment for this code block if you want to run Cross Validation instead, also comment out above code block ######

#         cross_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#              decision_function_shape='ovr', degree=3, gamma='auto',
#              kernel='rbf', max_iter=-1, probability=True, random_state=None,
#              shrinking=True, tol=0.001, verbose=False)
#         cross_model.fit(xc_train, yc_train)
        
#         # sort out the samples classified into the new class
#         for x in xc_test:
#             l = cross_model.predict(np.array([x]))
#             if l == np.array([10]):
#                 uu_count += 1
#                 unknown.append(list(x))
#                 unknown_label.append(10)

    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
    
    return unknown, unknown_label

    
def SVM_phase2(sample_size, x_train, x_test, y_train, y_test):
    
    sample_num = int(sample_size * y_test.shape[0])
    
    sample, sample_label, new_x_test, new_y_test             = random_sampling(sample_num, x_train, x_test, y_train, y_test)
            
    sample_label = [10 for i in range(len(sample_label))] 
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
   
    x_train2 = x_train
 
    print(un_class.shape)
    if un_class.shape[0] != 0:
        x_train2 = np.concatenate((x_train, un_class), axis=0)
    else:
        x_train2 = x_train
    y_train2 = np.append(y_train, un_label)

#     # train & test on the new sets
    model2 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model2.fit(x_train2, y_train2)
    score2 = model2.score(new_x_test, new_y_test)
    print("\nFinal accuracy score: ", score2)
    
    origin_pred = model2.predict(new_x_test)
    f_measure = f1_score(new_y_test, origin_pred, average='macro')
    print("Final F-measure: ", f_measure)
    
    return f_measure


def SVM_main(uu_mean, uu_cov, k, sample_size=0.3):
    x_train, x_test, y_train, y_test, J_1 = make_data(uu_mean, uu_cov, k)
    score1 = SVM_phase1(x_train, x_test, y_train, y_test)
    score2 = SVM_phase2(sample_size, x_train, x_test, y_train, y_test)
    return score1, score2, J_1


'''-------------------------------------------------------------------------'''
def uu_distance_test():
    
    dim = []
    benchmark =[]
    model = []
    S_b = []
    
    #iterating over dimensions     
    for i in range(1,101):

        known_mean = []
        
        #setting covariance for known classes
        known_cov_1 = [0.5*np.identity(i*3) for k in range(0,10)]
        
       
        loc_y = np.linspace(-20, 20, 10)
        known_mean=[]
        #setting mean for known classes
        for k in range(0,10):
            append = np.ones(i*3)
            append[1] = loc_y[k]
            known_mean.append(append)
        benchmark_i , final_accu_i_200, S_b_i_200 = SVM_main(known_mean, known_cov_1, i)
        dim.append(i*3)
        benchmark.append(benchmark_i)
        model.append(final_accu_i_200)
        S_b.append(S_b_i_200)
        
            

    plt.figure(1)
    plt.scatter(dim, model, marker='x', c='r')
    plt.scatter(dim, benchmark, marker='o', c='b')

    plt.title('Dimensionality vs Performance' )
    plt.xlabel('Dimensionality')
    plt.ylabel('F-measure')
    plt.legend(['RSTMD + SVM', 'Benchmark'], loc = 7)
   # plt.legend(['RSTCV + SVM', 'Benchmark'], loc = 7) 
    plt.savefig('Exp2-3.pdf')

    
uu_distance_test()

