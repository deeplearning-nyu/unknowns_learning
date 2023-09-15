# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:43:34 2020

@author: ZYCBl
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 20:46:13 2020

@author: ZYCBl
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import numpy as np
import torchvision
from torch.utils import data
from torchvision import datasets, models, transforms
import time
import os
import copy
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



def train_model(model, criterion, optimizer, scheduler,train_loader, num_epochs=25):
    since = time.time()
    
    for epoch in range(num_epochs):
        with open("run_info_par.txt","a") as f:
            f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
            f.write('-' * 10 + "\n")

        # Each epoch has a training and validation phase
        for phase in ['train']:
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            scheduler.step()

            epoch_loss = running_loss / dataset_sizes["train"]
            epoch_acc = running_corrects.double() / dataset_sizes["train"]
            with open("run_info_par.txt","a") as f:
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

    
        with open("run_info_par.txt","a") as f:
            f.write("\n")

    time_elapsed = time.time() - since
    with open("run_info_par.txt","a") as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.eval()
    return model



device_count = torch.cuda.device_count()
device_ids = list(range(device_count))
with open("run_info_par.txt","a") as f:
    f.write("This RUN use {} GPU\n".format(device_count))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

train_data_dir = './public_dataset/pytorch/svhn-data'

trainset = datasets.SVHN(root=train_data_dir, split='train', download=True, transform = in_transform)
testset = datasets.SVHN(root=train_data_dir, split='test', download=True, transform = in_transform)

X_train,y_train = trainset.data,trainset.labels
X_test,y_test = testset.data,testset.labels

to_hide = [9]
for label in to_hide:
    y_train = np.where(y_train == label, 99 ,y_train)
    y_test = np.where(y_test == label, 99, y_test)
    
X_train = X_train[y_train != 99]
y_train = y_train[y_train != 99]

y_test = np.where(y_test == 99, 9, y_test)

trainset.data,trainset.labels = None, None
testset.data,testset.labels = None, None

empty_train = copy.deepcopy(trainset)
empty_test = copy.deepcopy(testset)

trainset.data,trainset.labels = X_train,y_train
testset.data,testset.labels = X_test,y_test


# Test Sampling

sample,new_X_test, sample_label, new_y_test = train_test_split(X_test,y_test, test_size = 0.3, shuffle = True)
sample_label = 9 * np.ones_like(sample_label)
aug_X_train = np.concatenate([X_train, sample])
aug_y_train = np.concatenate([y_train,sample_label])

zipped = list(zip(aug_X_train,aug_y_train))  
random.shuffle(zipped)
aug_X_train,aug_y_train = zip(*zipped)
aug_X_train = np.array(aug_X_train)
aug_y_train = np.array(aug_y_train)


# Model construction
res34 = models.resnet34(pretrained = False)
class_to_train = 10
num_ftrs = res34.fc.in_features
res34.fc = nn.Linear(num_ftrs, class_to_train)


# Cross validation
kf = KFold(n_splits=3)
total_uu = []
for train_index, test_index in kf.split(aug_X_train, aug_y_train):
    # Generate dataset
    xc_train,yc_train = aug_X_train[train_index], aug_y_train[train_index]
    xc_test,yc_test = aug_X_train[test_index], aug_y_train[test_index]
    
    cv_train = copy.deepcopy(empty_train)
    cv_test = copy.deepcopy(empty_test)
    
    cv_train.data,cv_train.labels = xc_train, yc_train
    cv_test.data,cv_test.labels = xc_test,yc_test
    
    train_loader = data.DataLoader(cv_train, batch_size = 256, shuffle = True, pin_memory = True)
    test_loader = data.DataLoader(cv_test, batch_size = 256, shuffle = True, pin_memory = True)
    
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    learning_rate = 1e-4
    model = copy.deepcopy(res34)
    model = model.cuda()
    
    model = nn.DataParallel(model, device_ids = device_ids)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    dataset_sizes = {"train": len(trainset)}

    trained_model = train_model(model, criterion, optimizer, scheduler,train_loader, num_epochs=30)

    uu_X = []
    for data_batch in test_loader:    
        images, labels = data_batch
        images = images.cuda()
        labels = labels.cuda()
        
        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        images = images.cpu().detach().numpy()
        predicted = predicted.cpu().detach().numpy()
        this_uu = images[predicted == 9]
        
        uu_X.append(this_uu)
    uu_X = np.concatenate(uu_X)
    with open("run_info_par.txt","a") as f:
        f.write("In this split, we mined {} OOD".format(len(uu_X)))
    total_uu.append(uu_X)

total_uu = np.concatenate(total_uu)
total_uu_label = 9 * np.ones((len(total_uu),))


# Final Training

X_train = np.concatenate([X_train, total_uu])
y_train = np.concatenate([y_train, total_uu_label])

trainset.data,trainset.labels = X_train, y_train
testset.data
final_train = data.DataLoader(trainset, batch_size = 256, shuffle = True, pin_memory = True)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
learning_rate = 1e-4

model = copy.deepcopy(res34)
model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

final_model = train_model(model, criterion, optimizer, scheduler,train_loader, num_epochs=30)     


torch.save(final_model.state_dict(), "./model/svhn/SVHN_par.pth")

final_test = data.DataLoader(testset,batch_size = 256, shuffle = False, pin_memory = True)

predict = []
for data_batch in final_test:     # 每一个test mini-batch
    images, labels = data_batch
    images,labels = images.cuda(),labels.cuda()
    outputs = final_model(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu().detach().numpy()
    predict.append(predicted)

predict = np.concatenate(predict)
labels = testset.labels
with open("run_info_par.txt","a") as f:
    f.write("F1 SCORE {}\n".format(f1_score(labels, predict, average = "marcro")))
    f.write("F1 SCORE {}\n".format(accuracy_score(labels, predict)))















