 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wang, Zheng (zw1454@nyu.edu)
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


def uu_between_class_scatter_matrix(known_mean_list, unknown_mean):
    scatter_mat = np.zeros((2,2))
    for i in range(known_mean_list.shape[0]):
        mu_i = (known_mean_list[i] - unknown_mean).reshape(-1, 2)
        scatter_mat += np.matmul(mu_i.T, mu_i)
    
    return scatter_mat / (known_mean_list.shape[0] + 1)


def uu_mixture_scatter_matrix(known_mean_list, uu_known_mean, uu_cov_mat):
    S_b = uu_between_class_scatter_matrix(known_mean_list, uu_known_mean)
    
    return S_b + uu_cov_mat


def make_data(uu_mean, uu_cov):
    #set constant covariance and size for known classes
    cov = [[1.0, 0], [0, 1.0]]
    class_size = 100
    
    mean0 = [0, 7.0]
    mean1 = [0, 2.0]
    mean2 = [0, -2.0]
    mean3 = [0, -7.0]
    mean4 = [5.0, 7.0]
    mean5 = [5.0, 2.0]
    mean6 = [5.0, -2.0]
    mean7 = [5.0, -7.0]
    mean8 = [-5.0, 5.0]
    mean9 = [-5.0, -5.0]
    
    ################
    # compute the trace of scatter matrix
    known_mean_l = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9]
    known_mean_l = np.asarray(known_mean_l)
    uu_mean = np.asarray(uu_mean)
    
    trace_of_distance_scatter_matrix = np.trace(uu_between_class_scatter_matrix(known_mean_l, uu_mean))
    trace_of_mixture_scatter_matrix = np.trace(uu_mixture_scatter_matrix(known_mean_l, uu_mean, uu_cov))
    trace_of_cov_matrix = np.trace(uu_cov)
    print('\nTrace of between-class scatter matrix: ', trace_of_distance_scatter_matrix)
    print('Trace of mixture scatter matrix: ', trace_of_mixture_scatter_matrix)
    ################
    
    # construct the 10 known classes
    class0 = multivariate_normal(mean0, cov, class_size)
    class1 = multivariate_normal(mean1, cov, class_size)
    class2 = multivariate_normal(mean2, cov, class_size)
    class3 = multivariate_normal(mean3, cov, class_size)
    class4 = multivariate_normal(mean4, cov, class_size)
    class5 = multivariate_normal(mean5, cov, class_size)
    class6 = multivariate_normal(mean6, cov, class_size)
    class7 = multivariate_normal(mean7, cov, class_size)
    class8 = multivariate_normal(mean8, cov, class_size)
    class9 = multivariate_normal(mean9, cov, class_size)
    
    known_x = np.concatenate([class0, class1, class2, class3, class4, class5, class6,
                       class7, class8, class9])
    known_y = []
    for i in range(class_size * 10):
        known_y.append(i // class_size)
    known_y = np.array(known_y)
    
    # construct the unknown unknown class
    ux = multivariate_normal(uu_mean, uu_cov, 150)
    uy = np.array([10 for i in range(len(ux))])  
    
    all_x, all_y = np.concatenate((known_x, ux), axis=0), np.concatenate((known_y, uy), axis=0)
    
    x_train, x_test, y_train, y_test = train_test_split(known_x, known_y, test_size=0.3)
    x_test = np.concatenate((x_test, ux))
    y_test = np.concatenate((y_test, uy))
    
    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print("Shape of the original training data: ", x_train.shape)
    print("Shape of the original test data: ", x_test.shape)
    
#    plt.figure(1)
#    plt.scatter(all_x[:,0], all_x[:,1], c=all_y)
#    plt.xlim((-10, 10))
#    plt.ylim((-10, 10))
#    plt.title("The structure of the 10+uu classes")
#    plt.show()  # Note that the color of the unknown unknown class is yellow
    
    return x_train, x_test, y_train, y_test, \
        trace_of_cov_matrix, trace_of_mixture_scatter_matrix, trace_of_distance_scatter_matrix


def SVM_phase1(x_train, x_test, y_train, y_test):    
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model.fit(x_train, y_train)
    model.predict(x_test)
    score1 = model.score(x_test, y_test)
    print("\nOriginal accuracy score: ", score1)


def random_sampling(n, x_train, x_test, y_train, y_test):      
    # n: number of samples from the test set
    A = [list(i) for i in x_test]
    a = list(y_test) 
    newA, newa = zip(*random.sample(list(zip(A, a)), n))
    newA_l, newa_l = list(newA), list(newa)
    sample, sample_label = np.array(newA_l), np.array(newa_l) # obtain a sample of test set
    
    return sample, sample_label, x_test, y_test
   

def cross_validation(X, Y, k):
    # divide the training data into k-folds
    kf = KFold(n_splits=k)
    unknown = []
    unknown_label = []
    uu_count = 0
    
    for train_index, test_index in kf.split(X, Y):
        xc_train, xc_test = X[train_index], X[test_index]
        yc_train = Y[train_index]
        
        cross_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
             decision_function_shape='ovr', degree=3, gamma='auto',
             kernel='rbf', max_iter=-1, probability=True, random_state=None,
             shrinking=True, tol=0.001, verbose=False)
        cross_model.fit(xc_train, yc_train)
        
        # sort out the samples classified into the new class
        for x in xc_test:
            l = cross_model.predict(np.array([x]))
            if l == np.array([10]):
                uu_count += 1
                unknown.append(list(x))
                unknown_label.append(10)

    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
    
    return unknown, unknown_label

    
def SVM_phase2(sample_size, x_train, x_test, y_train, y_test, k):
    sample, sample_label, new_x_test, new_y_test \
            = random_sampling(sample_size, x_train, x_test, y_train, y_test)
            
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
    
    un_class, un_label = cross_validation(classes, classes_label, k)
    
    # construct the final training set by adding the unknown unknowns
    if un_class.shape[0] != 0:
        x_train2 = np.concatenate((x_train, un_class), axis=0)
    else:
        x_train2 = x_train
    y_train2 = np.append(y_train, un_label)

    # train & test on the new sets
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
    
    return score2


def SVM_main(uu_mean, uu_cov, sample_size=60, k=10):
    x_train, x_test, y_train, y_test, trace1, trace2, trace3 = make_data(uu_mean, uu_cov)
    SVM_phase1(x_train, x_test, y_train, y_test)
    score2 = SVM_phase2(sample_size, x_train, x_test, y_train, y_test, k)
    return score2, trace1, trace2, trace3


'''-------------------------------------------------------------------------'''
def uu_distance_test():
    uu_cov = [[1.5, 0], [0, 1.5]]
    # create list of locations of uu along y = 0
    loc_on_x_axis = np.linspace(-10, 10, 100)
    
    trace_l = []
    final_accu_l = []
    
    for i in range(loc_on_x_axis.shape[0]):
        uu_mean_i = [loc_on_x_axis[i], 0]
        final_accu_i, _, _, trace_i = SVM_main(uu_mean_i, uu_cov)
        trace_l.append(trace_i)
        final_accu_l.append(final_accu_i)
        
    plt.figure(2)
    plt.scatter(trace_l, final_accu_l, marker='x', c='red')
    plt.hlines(0.66, 35, 150)
    plt.xlabel('Trace of between-class scatter matrix')
    plt.ylabel('F-measure')
    plt.legend(['RTSCV+SVM', 'Benchmark'], loc=7)
    plt.savefig('uu_distance_separability.pdf')
    
    
def uu_cov_test():
    uu_mean = [-4, 0]
    uu_cov_l = np.linspace(0, 20, 40)
    
    trace_l = []
    final_accu_l = []
    
    for i in range(uu_cov_l.shape[0]):
        uu_cov_i = np.eye(2) * uu_cov_l[i]
        final_accu_i, trace_i, _, _ = SVM_main(uu_mean, uu_cov_i)
        trace_l.append(trace_i)
        final_accu_l.append(final_accu_i)
        
    plt.figure(3)
    plt.scatter(trace_l, final_accu_l, marker='x', c='red')
    plt.hlines(0.66, 59, 100)
    plt.xlabel('Trace of mixture scatter matrix')
    plt.ylabel('F-measure')
    plt.legend(['RTSCV+SVM', 'Benchmark'])
    plt.savefig('uu_cov_separability.pdf')
    

def uu_J1_test():
    loc_on_x_axis = np.linspace(-10, 10, 20)
    uu_cov_l = np.array([1, 2, 4, 8, 16])
    
    plt.figure(5)
    marker = ['.', '1', '*', 'x', '4']
    for j in range(uu_cov_l.shape[0]):
        J1_l = []
        final_accu_l = []
        for i in range(loc_on_x_axis.shape[0]):
            uu_mean = [loc_on_x_axis[i], 0]
            uu_cov = np.eye(2) * uu_cov_l[j]
            final_accu, trace1, trace2, _ = SVM_main(uu_mean, uu_cov)
            J1_l.append(trace2/trace1)
            final_accu_l.append(final_accu)
        plt.scatter(J1_l, final_accu_l, marker=marker[j])
        
    plt.hlines(0.66, 0, 80)
    plt.xlabel('J1 score')
    plt.ylabel('F-measure')
    plt.legend(['cov=1', 'cov=2', 'cov=4', 'cov=8', 'cov=16', 'benchmark'])
    plt.savefig('uu_distance_separability.pdf')
    
    
