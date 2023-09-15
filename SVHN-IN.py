# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:59:14 2021

@author: ZYCBl
"""
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


class customDataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        to_return_data = self.data[idx]
        to_return_label = self.target[idx]
        return (to_return_data,to_return_label)

def test_model(final_model,testloader):
    predict = []

    for data_batch in testloader:     # 每一个test mini-batch
        images, labels = data_batch
        images,labels = images.cuda(),labels.cuda()
        outputs = final_model(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().detach().numpy()
        predict.append(predicted)
        
    predict = np.concatenate(predict)
    predict = np.array(predict, dtype = np.int64)
    return predict

def train_model(model, criterion, optimizer, scheduler,train_loader, num_epochs=25):
    since = time.time()
    
    for epoch in range(num_epochs):
        with open("SVHN-IN-INFO.txt","a") as f:
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
            with open("SVHN-IN-INFO.txt","a") as f:
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

    
        with open("SVHN-IN-INFO.txt","a") as f:
            f.write("\n")

    time_elapsed = time.time() - since
    with open("SVHN-IN-INFO.txt","a") as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.eval()
    return model



device_count = torch.cuda.device_count()
device_ids = list(range(device_count))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("This RUN use {} GPU\n".format(device_count))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

alter_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

train_data_dir = './public_dataset/pytorch/svhn-data'
imagenet_dir = './public_dataset/pytorch/Imagenet_resize'
trainset = datasets.SVHN(root=train_data_dir, split='train', download=True, transform = in_transform)
testset = datasets.SVHN(root=train_data_dir, split='test', download=True, transform = in_transform)

origin_train_loader = data.DataLoader(trainset, batch_size = len(trainset), shuffle = True, pin_memory = True)
origin_test_loader = data.DataLoader(testset, batch_size = len(testset), shuffle = True, pin_memory = True)

imagenet = torchvision.datasets.ImageFolder(root=imagenet_dir, transform = alter_transform)
imagenet_loader = data.DataLoader(imagenet,batch_size = len(imagenet),shuffle = True)

imagenet_data, imagenet_target = next(iter(imagenet_loader))
imagenet_data = imagenet_data.cpu().detach().numpy()
imagenet_labels = 10 * np.ones((len(imagenet_data),))

X_train,y_train = next(iter(origin_train_loader))
y_train = np.array(y_train, dtype = np.int64)

X_test,y_test = next(iter(origin_test_loader))

if torch.is_tensor(X_train):
    X_train = X_train.cpu().detach().numpy()
if torch.is_tensor(y_train):
    y_train = y_train.cpu().detach().numpy()
if torch.is_tensor(X_test):
    X_test = X_test.cpu().detach().numpy()
if torch.is_tensor(y_test):
    y_t = y_test.cpu().detach().numpy()    
    

new_X_test = np.concatenate([X_test,imagenet_data])
new_y_test = np.concatenate([y_test,imagenet_labels])
new_y_test = np.array(new_y_test, dtype = np.int64)

trainset = customDataset(X_train,y_train)
origin_train_loader = data.DataLoader(trainset, batch_size = len(trainset), shuffle = True, pin_memory = True)

# Test Sampling

new_X,sample,new_y, sample_label = train_test_split(new_X_test,new_y_test, test_size = 0.15, shuffle = True)

sample_label = 10 * np.ones_like(sample_label)
aug_X_train = np.concatenate([X_train, sample])
aug_y_train = np.concatenate([y_train,sample_label])

aug_y_train = np.array(aug_y_train, dtype = np.int64)

zipped = list(zip(aug_X_train,aug_y_train))  
random.shuffle(zipped)
aug_X_train,aug_y_train = zip(*zipped)
aug_X_train = np.array(aug_X_train)
aug_y_train = np.array(aug_y_train)


# Model construction
res34 = models.resnet34(pretrained = False)
class_to_train = 11
num_ftrs = res34.fc.in_features
res34.fc = nn.Linear(num_ftrs, class_to_train)


# Cross validation
kf = KFold(n_splits=3)
total_uu = []
for train_index, test_index in kf.split(aug_X_train, aug_y_train):
    # Generate dataset
    xc_train,yc_train = aug_X_train[train_index], aug_y_train[train_index]
    xc_test,yc_test = aug_X_train[test_index], aug_y_train[test_index]
    
    cv_train = customDataset(xc_train, yc_train)
    cv_test = customDataset(xc_test,yc_test)
    
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
    
    dataset_sizes = {"train": len(cv_train)}

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
        this_uu = images[predicted == 10]
        
        uu_X.append(this_uu)
    uu_X = np.concatenate(uu_X)
    
    with open("SVHN-IN-INFO.txt","a") as f:
        f.write("In this split, we mined {} OOD\n".format(len(uu_X)))
    total_uu.append(uu_X)


total_uu = np.concatenate(total_uu)
total_uu_label = 10 * np.ones((len(total_uu),))
# Final Training
X_train, y_train = next(iter(origin_train_loader))

X_train = X_train.cpu().detach().numpy()
y_train = y_train.cpu().detach().numpy()

rect_X_train = np.concatenate([X_train, total_uu])
rect_y_train = np.concatenate([y_train, total_uu_label])
rect_y_train = np.array(rect_y_train, dtype = np.int64)


reduced_uu = total_uu.reshape((total_uu.shape[0],-1))
raw_train = X_train.reshape((X_train.shape[0],-1))
uu_rows = reduced_uu.view([('', reduced_uu.dtype)] * reduced_uu.shape[1])
train_rows = raw_train.view([('', raw_train.dtype)] * raw_train.shape[1])
reduced_uu = np.setdiff1d(uu_rows, train_rows).view(total_uu.dtype).reshape(-1,total_uu.shape[1])

reduced_uu = reduced_uu.reshape((-1,3, 32,32))
reduced_uu_label = 10 * np.ones((len(total_uu),))

reduced_X_train = np.concatenate([X_train, reduced_uu])
reduced_y_train = np.concatenate([y_train, reduced_uu_label])
reduced_y_train = np.array(reduced_y_train, dtype = np.int64)


pre_rect_trainset = customDataset(X_train,y_train)
finaltrainset = customDataset(rect_X_train,rect_y_train)
reduced_trainset = customDataset(reduced_X_train, reduced_y_train)

IDtest = customDataset(X_test,y_test)
OODtest = customDataset(imagenet_data,imagenet_labels)
Hybridtest = customDataset(new_X_test, new_y_test)

origin_train = data.Dataloader(pre_rect_trainset, batch_size = 256, shuffle = True, pin_memory = True)
final_train = data.DataLoader(finaltrainset, batch_size = 256, shuffle = True, pin_memory = True)
reduced_train = data.DataLoader(reduced_trainset, batch_size = 256, shuffle = True, pin_memory = True)


IDtest_loader = data.DataLoader(IDtest, batch_size = 256, shuffle = False, pin_memory = True)
OODtest_loader = data.DataLoader(OODtest, batch_size = 256, shuffle = False, pin_memory = True)
Hybridtest_loader = data.DataLoader(Hybridtest, batch_size = 256, shuffle = False, pin_memory = True)



# Rectified(No correction)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
learning_rate = 1e-4
model = copy.deepcopy(res34)
model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
dataset_sizes = {"train": len(finaltrainset)}
final_model = train_model(model, criterion, optimizer, scheduler,final_train, num_epochs=30)     
torch.save(final_model.state_dict(), "./model/svhn/SVHN-IN-INFO.pth")

IDpredict = test_model(final_model, IDtest_loader)
IDlabels = y_test
print("RECTIFIED ID LABELS\n")
print(Counter(IDpredict))

print(Counter(y_test.cpu().detach().numpy()))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("Rectified with No Correction \n")
    f.write("ID F1 SCORE {}\n".format(f1_score(IDlabels, IDpredict, average = "macro")))
    f.write("ID Accuracy {}\n".format(accuracy_score(IDlabels, IDpredict)))


OODpredict = test_model(final_model, OODtest_loader)
OODlabels = imagenet_labels
print(Counter(OODlabels))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("OOD Accuracy {}\n".format(accuracy_score(OODlabels, OODpredict)))


Hybridpredict = test_model(final_model, Hybridtest_loader)
Hybridlabels = new_y_test
print(Counter(Hybridlabels))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("Hybrid F1 SCORE {}\n".format(f1_score(Hybridlabels, Hybridpredict, average = "macro")))
    f.write("Hybrid Accuracy {}\n".format(accuracy_score(Hybridlabels, Hybridpredict)))
    f.write("-" *10 + "\n")


# Pre-rectified
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
learning_rate = 1e-4
model = copy.deepcopy(res34)
model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
dataset_sizes = {"train": len(pre_rect_trainset)}

final_model = train_model(model, criterion, optimizer, scheduler,origin_train, num_epochs=30)     
torch.save(final_model.state_dict(), "./model/svhn/SVHN-IN-INFO-pre-rect.pth")

IDpredict = test_model(final_model, IDtest_loader)
print("PRE-RECTIFIED ID LABELS\n")
print(Counter(IDpredict))
IDlabels = y_test
print(Counter(y_test.cpu().detach().numpy()))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("Pre-rectified \n")
    f.write("ID F1 SCORE {}\n".format(f1_score(IDlabels, IDpredict, average = "macro")))
    f.write("ID Accuracy {}\n".format(accuracy_score(IDlabels, IDpredict)))


OODpredict = test_model(final_model, OODtest_loader)
OODlabels = imagenet_labels
print(Counter(OODlabels))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("OOD Accuracy {}\n".format(accuracy_score(OODlabels, OODpredict)))


Hybridpredict = test_model(final_model, Hybridtest_loader)
Hybridlabels = new_y_test
print(Counter(Hybridlabels))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("Hybrid F1 SCORE {}\n".format(f1_score(Hybridlabels, Hybridpredict, average = "macro")))
    f.write("Hybrid Accuracy {}\n".format(accuracy_score(Hybridlabels, Hybridpredict)))


# Rectified(With correction)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
learning_rate = 1e-4
model = copy.deepcopy(res34)
model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
dataset_sizes = {"train": len(reduced_trainset)}

final_model = train_model(model, criterion, optimizer, scheduler,reduced_train, num_epochs=30)     
torch.save(final_model.state_dict(), "./model/svhn/SVHN-IN-INFO-reduced.pth")

IDpredict = test_model(final_model, IDtest_loader)
IDlabels = y_test
print("RECTIFIED WITH CORRECTION ID LABELS\n")
print(Counter(IDpredict))

print(Counter(y_test.cpu().detach().numpy()))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("Rectified with No Correction \n")
    f.write("ID F1 SCORE {}\n".format(f1_score(IDlabels, IDpredict, average = "macro")))
    f.write("ID Accuracy {}\n".format(accuracy_score(IDlabels, IDpredict)))


OODpredict = test_model(final_model, OODtest_loader)
OODlabels = imagenet_labels
print(Counter(OODlabels))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("OOD Accuracy {}\n".format(accuracy_score(OODlabels, OODpredict)))


Hybridpredict = test_model(final_model, Hybridtest_loader)
Hybridlabels = new_y_test
print(Counter(Hybridlabels))
with open("SVHN-IN-INFO.txt","a") as f:
    f.write("Hybrid F1 SCORE {}\n".format(f1_score(Hybridlabels, Hybridpredict, average = "macro")))
    f.write("Hybrid Accuracy {}\n".format(accuracy_score(Hybridlabels, Hybridpredict)))






