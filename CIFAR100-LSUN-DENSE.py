# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:00:13 2021

@author: ZYCBl
"""
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
from resnet import ResNet34
from densenet_MD import DenseNet3
import torch
import torch.nn as nn
from torch.nn.functional import softmax
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

NUM_WORKERS = 0
learning_rate = 5e-4

def compute_auc(label, scores):
    label = np.where(label == 100, 1, 0)
    auroc = roc_auc_score(label, scores)
    precision, recall, _ = precision_recall_curve(label,scores)
    aupr = auc(recall,precision)
    
    return auroc, aupr

def get_resnet34():
    res34 = ResNet34(100)
    res34.load_state_dict(torch.load("pretrained/resnet_cifar100.pth"))
    class_to_train = 101
    num_ftrs = res34.linear.in_features
    res34.linear = nn.Linear(num_ftrs, class_to_train)
    
    return res34

def get_densenet():
    densenet = DenseNet3(100,100)
    densenet.load_state_dict(torch.load("pretrained/densenet_cifar100.pth"))
    class_to_train = 101
    num_ftrs = densenet.fc.in_features
    densenet.fc = nn.Linear(num_ftrs, class_to_train)
    
    return densenet
def save_scores(label,scores, stype):
    label = np.where(label == 100, 1, 0)
    known = scores[label == 0]
    novel = scores[label == 1]
    
    with open("CIFAR100-LSUN-DENSE-known_scores_{}".format(stype), "a") as f:
        for _ in known:
            f.write("{:7.4f}\n".format(_))
            
    with open("CIFAR100-LSUN-DENSE-novel_scores_{}".format(stype), "a") as f:
        for _ in novel:
            f.write("{:7.4f}\n".format(_))
            
            
            
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
    final_model.eval()
    predict = []
    origin = []
    scores = []
    for data_batch in testloader:     # 每一个test mini-batch
        images, labels = data_batch
        images,labels = images.cuda(),labels.cuda()
        outputs = final_model(images)
        score = softmax(outputs.data, dim = 1)[:,100]
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().detach().numpy()
        score = score.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        origin.append(labels)
        predict.append(predicted)
        scores.append(score)
        
    origin = np.concatenate(origin) 
    predict = np.concatenate(predict)
    scores = np.concatenate(scores)
    origin = np.array(origin,dtype = np.int64)
    predict = np.array(predict, dtype = np.int64)
    return predict, origin,scores

def train_model(model, criterion, optimizer, scheduler,train_loader,test_loader, num_epochs = 60):
    since = time.time()
    
    best_model = None
    best_ac = - float("inf")
    for epoch in range(num_epochs):
        model.train()
        with open("load.txt","a") as f:
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
                    
            with open("load.txt","a") as f:
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

        # In Evaluation Phase
        predicted, origin,_ = test_model(model,test_loader)
        validation_ac = accuracy_score(origin, predicted)
        if validation_ac > best_ac:
            best_ac = validation_ac
            best_model = copy.deepcopy(model)
        with open("load.txt","a") as f:
                f.write('{} Acc: {:.4f}\n'.format("EVAL", validation_ac))

    
        # with open("SVHN-LSUN.txt","a") as f:
        #     f.write("\n")

    time_elapsed = time.time() - since
    with open("load.txt","a") as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.eval()
    best_model.eval()
    return best_model



device_count = torch.cuda.device_count()
device_ids = list(range(device_count))
with open("load.txt","a") as f:
    f.write("This RUN use {} GPU\n".format(device_count))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])

train_data_dir = './public_dataset/pytorch/CIFAR100'
imagenet_dir = './public_dataset/pytorch/LSUN_resize'
trainset = datasets.CIFAR100(root=train_data_dir, train = True, download=True, transform = in_transform)
testset = datasets.CIFAR100(root=train_data_dir, train = False, download=True, transform = in_transform)

origin_train_loader = data.DataLoader(trainset, batch_size = len(trainset), shuffle = True, pin_memory = True, num_workers = NUM_WORKERS, drop_last=True)
origin_test_loader = data.DataLoader(testset, batch_size = len(testset), shuffle = True, pin_memory = True, num_workers = NUM_WORKERS, drop_last=True)

imagenet = torchvision.datasets.ImageFolder(root=imagenet_dir, transform = in_transform)
imagenet_loader = data.DataLoader(imagenet,batch_size = len(imagenet),shuffle = True,num_workers = NUM_WORKERS, drop_last=True)

imagenet_data, imagenet_target = next(iter(imagenet_loader))
imagenet_data = imagenet_data.cpu().detach().numpy()
imagenet_labels = 100 * np.ones((len(imagenet_data),))

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
origin_train_loader = data.DataLoader(trainset, batch_size = len(trainset), shuffle = True, pin_memory = True,num_workers = NUM_WORKERS, drop_last=True)

# Test Sampling
sr_lst = [0.06]
for rate in sr_lst:
    new_X,sample,new_y, sample_label = train_test_split(new_X_test,new_y_test, test_size = rate, shuffle = True)
    
    sample_label = 100 * np.ones_like(sample_label)
    aug_X_train = np.concatenate([X_train, sample])
    aug_y_train = np.concatenate([y_train,sample_label])
    
    aug_y_train = np.array(aug_y_train, dtype = np.int64)
    
    zipped = list(zip(aug_X_train,aug_y_train))  
    random.shuffle(zipped)
    aug_X_train,aug_y_train = zip(*zipped)
    aug_X_train = np.array(aug_X_train)
    aug_y_train = np.array(aug_y_train)
    
    
    # Model construction

    
    
    # Cross validation
    kf = KFold(n_splits=3)
    total_uu = []
    for train_index, test_index in kf.split(aug_X_train, aug_y_train):
        # Generate dataset
        xc_train,yc_train = aug_X_train[train_index], aug_y_train[train_index]
        xc_test,yc_test = aug_X_train[test_index], aug_y_train[test_index]
        
        cv_train = customDataset(xc_train, yc_train)
        cv_test = customDataset(xc_test,yc_test)
        
        train_loader = data.DataLoader(cv_train, batch_size = 256, shuffle = True, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
        test_loader = data.DataLoader(cv_test, batch_size = 256, shuffle = True, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
        
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        model = get_densenet()
        model = model.cuda()
        
        model = nn.DataParallel(model, device_ids = device_ids)
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        dataset_sizes = {"train": len(cv_train)}
    
        trained_model = train_model(model, criterion, optimizer, scheduler,train_loader,test_loader)
    
        uu_X = []
        for data_batch in test_loader:    
            images, labels = data_batch
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            images = images.cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            this_uu = images[predicted == 100]
            
            uu_X.append(this_uu)
        uu_X = np.concatenate(uu_X)
        
        with open("load.txt","a") as f:
            f.write("In this split, we mined {} OOD\n".format(len(uu_X)))
        total_uu.append(uu_X)
    
    
    total_uu = np.concatenate(total_uu)
    total_uu_label = 100 * np.ones((len(total_uu),))
    # Final Training
    X_train, y_train = next(iter(origin_train_loader))
    
    X_train = X_train.cpu().detach().numpy()
    y_train = y_train.cpu().detach().numpy()
    
    rect_X_train = np.concatenate([X_train, total_uu])
    rect_y_train = np.concatenate([y_train, total_uu_label])
    rect_y_train = np.array(rect_y_train, dtype = np.int64)
    
    
    # reduced_uu = total_uu.reshape((total_uu.shape[0],-1))
    # raw_train = X_train.reshape((X_train.shape[0],-1))
    # uu_rows = reduced_uu.view([('', reduced_uu.dtype)] * reduced_uu.shape[1])
    # train_rows = raw_train.view([('', raw_train.dtype)] * raw_train.shape[1])
    # reduced_uu = np.setdiff1d(uu_rows, train_rows).view(total_uu.dtype).reshape(-1,total_uu.shape[1])
    
    # reduced_uu = reduced_uu.reshape((-1,3, 32,32))
    # reduced_uu_label = 100 * np.ones((len(total_uu),))
    
    # reduced_X_train = np.concatenate([X_train, reduced_uu])
    # reduced_y_train = np.concatenate([y_train, reduced_uu_label])
    # reduced_y_train = np.array(reduced_y_train, dtype = np.int64)
    
    
    pre_rect_trainset = customDataset(X_train,y_train)
    finaltrainset = customDataset(rect_X_train,rect_y_train)
    # reduced_trainset = customDataset(reduced_X_train, reduced_y_train)
    
    IDtest = customDataset(X_test,y_test)
    OODtest = customDataset(imagenet_data,imagenet_labels)
    Hybridtest = customDataset(new_X_test, new_y_test)
    
    origin_train = data.DataLoader(pre_rect_trainset, batch_size = 256, shuffle = True, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
    final_train = data.DataLoader(finaltrainset, batch_size = 256, shuffle = True, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
    # reduced_train = data.DataLoader(reduced_trainset, batch_size = 256, shuffle = True, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
    
    
    IDtest_loader = data.DataLoader(IDtest, batch_size = 256, shuffle = False, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
    OODtest_loader = data.DataLoader(OODtest, batch_size = 256, shuffle = False, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
    Hybridtest_loader = data.DataLoader(Hybridtest, batch_size = 256, shuffle = False, pin_memory = True,num_workers = NUM_WORKERS,drop_last=True)
    
    
    
    # Rectified(No correction)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = get_densenet()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids = device_ids)
    optimizer = optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    dataset_sizes = {"train": len(finaltrainset)}
    final_model = train_model(model, criterion, optimizer, scheduler,final_train, Hybridtest_loader)     
    # torch.save(final_model.state_dict(), "./model/CIFAR100/sample_rate.pth")
    
    IDpredict,IDlabels,ID_scores = test_model(final_model, IDtest_loader)
    counter = Counter(IDpredict)
    ratio = counter[100]/len(IDpredict)
    
    OODpredict,OODlabels,OOD_scores = test_model(final_model, OODtest_loader)
    Hybridpredict,Hybridlabels, Hybrid_scores = test_model(final_model, Hybridtest_loader)

    ac1 = accuracy_score(IDlabels, IDpredict)
    ac2 = accuracy_score(OODlabels, OODpredict)
    ac3 = accuracy_score(Hybridlabels, Hybridpredict)
    auroc,aupr = compute_auc(Hybridlabels, Hybrid_scores)
    save_scores(Hybridlabels, Hybrid_scores, "Rectify")
    with open("CIFAR100-LSUN-DENSE.txt","a") as f:
        f.write("{:7.2f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\n".format(rate, ac1, ac2, ac3,ratio, auroc,aupr))
    
        
    # Pre-rectified
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = get_densenet()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids = device_ids)
    optimizer = optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    dataset_sizes = {"train": len(pre_rect_trainset)}
    
    final_model = train_model(model, criterion, optimizer, scheduler,origin_train, Hybridtest_loader, num_epochs = 60)    
    # torch.save(final_model.state_dict(), "./model/CIFAR100/sample_rate-pre-rect.pth")
    
    IDpredict,IDlabels,ID_scores = test_model(final_model, IDtest_loader)
    counter = Counter(IDpredict)
    ratio = counter[100]/len(IDpredict)
    OODpredict,OODlabels,OOD_scores = test_model(final_model, OODtest_loader)
    
    Hybridpredict,Hybridlabels,Hybrid_scores = test_model(final_model, Hybridtest_loader)
    
  
    ac1 = accuracy_score(IDlabels, IDpredict)
    ac2 = accuracy_score(OODlabels, OODpredict)
    ac3 = accuracy_score(Hybridlabels, Hybridpredict)
    auroc,aupr = compute_auc(Hybridlabels, Hybrid_scores)
    save_scores(Hybridlabels, Hybrid_scores, "Pre-Rectify")
    with open("CIFAR100-LSUN-DENSE.txt","a") as f:
        f.write("{:7.2f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\n".format(rate, ac1, ac2, ac3,ratio, auroc,aupr))
        
    # # Rectified(With correction)
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    # model = get_densenet()
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids = device_ids)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # dataset_sizes = {"train": len(reduced_trainset)}
    
    # final_model = train_model(model, criterion, optimizer, scheduler,reduced_train, Hybridtest_loader)     
    # torch.save(final_model.state_dict(), "./model/CIFAR100/sample_rate-reduced.pth")
    
    # IDpredict,IDlabels, ID_scores = test_model(final_model, IDtest_loader)
    # counter = Counter(IDpredict)
    # ratio = counter[100]/len(IDpredict)
    # OODpredict,OODlabels, OOD_scores = test_model(final_model, OODtest_loader)
    
    # Hybridpredict,Hybridlabels, Hybrid_scores = test_model(final_model, Hybridtest_loader)
   
    # ac1 = accuracy_score(IDlabels, IDpredict)
    # ac2 = accuracy_score(OODlabels, OODpredict)
    # ac3 = accuracy_score(Hybridlabels, Hybridpredict)
    # auroc,aupr = compute_auc(Hybridlabels, Hybrid_scores)
    # with open("CIFAR100-LSUN-DENSE.txt","a") as f:
    #     f.write("{:7.2f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\t{:7.4f}\n".format(rate, ac1, ac2, ac3,ratio, auroc,aupr))
        

