#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 9 13:07:12 2020

@author: Zheng Wang (zw1454@nyu.edu)
"""
import random
import numpy as np
import keras
from scipy.io import loadmat
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization

uu_label = []
uu_label_test = 10

input_shape = (32, 32, 3)
epochs = 30


def make_data():
    # Load the data
    train_raw = loadmat('train_32x32.mat')
    test_raw = loadmat('test_32x32.mat')
    
    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])
    train_labels = train_raw['y']
    test_labels = test_raw['y']
    
    # Fix the axes of the images
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)
    
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_labels = train_labels.astype('int32') - 1
    test_labels = test_labels.astype('int32') - 1
        
    # Normalize the images data
    train_images /= 255.0
    test_images /= 255.0
    
    old_x_train, old_y_train, x_test, y_test = train_images, train_labels, test_images, test_labels
    
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


def CNN_phase1(x_train, x_test, y_train, y_test):    
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=8,
                        zoom_range=[0.95, 1.05],
                        height_shift_range=0.10,
                        shear_range=0.15)
    
    # Define CNN model
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),    
        Dense(uu_label_test+1,  activation='softmax')
    ])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=x_train.shape[0] // 128,
                        epochs=epochs, validation_data=(x_test, y_test),
                        callbacks=[early_stopping])
    
    score1 = model.evaluate(x_test, y_test)[1]
    print("\nOriginal accuracy score: ", score1)
    
    origin_pred = model.predict_classes(x_test)
    f_measure = f1_score(y_test, origin_pred, average='macro')
    print("\nOriginal F-measure: ", f_measure)
    
    return (score1, f_measure)
    

def random_sampling(n, x_train, x_test, y_train, y_test):      
    # n: number of samples from the test set
    A = [list(i) for i in x_test]
    a = list(y_test) 
    newA, newa = zip(*random.sample(list(zip(A, a)), n))
    newA_l, newa_l = list(newA), list(newa)
    sample, sample_label = np.array(newA_l), np.array(newa_l) # obtain a sample of test set

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
        
        # Data augmentation
        datagen = ImageDataGenerator(rotation_range=8,
                            zoom_range=[0.95, 1.05],
                            height_shift_range=0.10,
                            shear_range=0.15)
        
        # Define CNN model
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            BatchNormalization(),
            
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.4),    
            Dense(uu_label_test+1,  activation='softmax')
        ])
        
        early_stopping = keras.callbacks.EarlyStopping(patience=8)
        optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit_generator(datagen.flow(xc_train, yc_train, batch_size=128),
                            steps_per_epoch=xc_train.shape[0] // 128,
                            epochs=epochs, validation_data=(xc_test, yc_test),
                            callbacks=[early_stopping])

        # sort out the samples classified into the new class
        for x in xc_test:
            l = model.predict_classes(np.array([x]))
            if l == np.array([uu_label_test]):                                    ####
                uu_count += 1
                unknown.append(list(x))
                unknown_label.append(uu_label_test)                               #####
    
    print("\nThe number of elements in the unknown unknown class: ", uu_count)
    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
    
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
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=8,
                        zoom_range=[0.95, 1.05],
                        height_shift_range=0.10,
                        shear_range=0.15)
    
    # Define CNN model
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),    
        Dense(uu_label_test+1,  activation='softmax')
    ])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    model.fit_generator(datagen.flow(x_train2, y_train2, batch_size=128),
                        steps_per_epoch=x_train.shape[0] // 128,
                        epochs=epochs, validation_data=(new_x_test, new_y_test),
                        callbacks=[early_stopping])
    
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