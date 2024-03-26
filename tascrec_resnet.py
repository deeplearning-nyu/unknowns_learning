import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset 
from torch.optim import lr_scheduler, SGD
from torchvision.datasets import ImageFolder
import numpy as np
import copy
import json, random
import os, time 
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score,\
                            roc_curve, precision_recall_curve, auc


openood_path = './OpenOOD'
sys.path.append(openood_path)
from openood.networks import ResNet18_32x32, ResNet18_224x224 
import openood.utils 
from openood.datasets import get_dataloader, get_ood_dataloader



''' We work with the field 'targets' in datasets.
    The following classes align the dataset types to
    have the field '''
# To create ood dataset with the same structure as the ID dataset
# and the same ood targets for all images
class ImglistDatasetWrapper(Dataset):
    def __init__(self, imglist_dataset, targets):
        self.dataset = imglist_dataset
        self.classes = imglist_dataset.num_classes
        self.targets = targets 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]['data'] 
        target = self.targets[idx]
        return data, target

# A VisionDataset that updates labels in 'targets'
# and keeps the old labels for analysis 
class HistoricVisionDataset(torch.utils.data.Dataset):
    def __init__(self, concat_dataset, new_targets):
        self.concat_dataset = concat_dataset
        self.targets = new_targets 

    def __getitem__(self, index):
        data, _ = self.concat_dataset[index]
        target = self.targets[index]
        _, old_target = self.concat_dataset[index]
        return data, target, old_target

    def __len__(self):
        return len(self.concat_dataset)

# 'targets' in an ImageFolder instance is just a convinient copy
# Fix the ImageFolder class to return the 'targets',
# not the labels in each sample 
class CustomImageFolder(ImageFolder):
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        target = self.targets[idx]
        return img, target
    
def get_config(**config):
    if config.get('ID', 'cifar10').lower() == 'cifar10':
        # load config files for cifar10 baseline
        config_files = [
            'OpenOOD/configs/datasets/cifar10/cifar10.yml',
            'OpenOOD/configs/datasets/cifar10/cifar10_ood.yml',
            'OpenOOD/configs/networks/resnet18_32x32.yml',
            'OpenOOD/configs/pipelines/test/test_ood.yml',
            'OpenOOD/configs/preprocessors/base_preprocessor.yml',
            'OpenOOD/configs/postprocessors/msp.yml'
        ]
    elif config.get('ID', 'cifar10').lower() == 'cifar100': 
        config_files = [
            'OpenOOD/configs/datasets/cifar100/cifar100.yml',
            'OpenOOD/configs/datasets/cifar100/cifar100_ood.yml',
            'OpenOOD/configs/networks/resnet18_32x32.yml',
            'OpenOOD/configs/pipelines/test/test_ood.yml',
            'OpenOOD/configs/preprocessors/base_preprocessor.yml',
            'OpenOOD/configs/postprocessors/msp.yml'
        ]
    elif config.get('ID', 'cifar10').lower() == 'imagenet200': 
        config_files = [
            'OpenOOD/configs/datasets/imagenet200/imagenet200.yml',
            'OpenOOD/configs/datasets/imagenet200/imagenet200_ood.yml',
            'OpenOOD/configs/networks/resnet18_224x224.yml',
            'OpenOOD/configs/pipelines/test/test_ood.yml',
            'OpenOOD/configs/preprocessors/base_preprocessor.yml',
            'OpenOOD/configs/postprocessors/msp.yml'
        ]
    else:
        raise ValueError("ID dataset not implemented") 
    
    return openood.utils.config.Config(*config_files) 

  
def mkdirs():
    directories = ['runs', 'runs/logs', 'runs/data_cache', 'runs/models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# print to stdout and file
def pw(file, msg):
    print(msg)
    with open(file, "a") as log_file:
        log_file.write(msg + '\n')

# returns the pre-trained CIFAR-10 ResNet-18 model
def get_resnet18(**config):
    
    log = config.get('log_filename',
                     "runs/logs/run_tascrec_noconfig.txt")
    ID = config.get('ID', 'CIFAR10').lower()
    
    model_file_dict = {
        'cifar10': 
            ['results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt', 10],
        'cifar100':
            ['results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt', 100],
        'imagenet200' : 
            ['results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1/best.ckpt', 200],
    }
    
    ID_model_file= model_file_dict[ID][0] 
    ID_classes = model_file_dict[ID][1]
    
    if not os.path.exists(ID_model_file):
        raise ValueError("Model not found, run OpenOOD/scripts/download/download.sh")
    else: 
        pw(log, "Model already downloaded.")

    # load the model
    if ID == 'imagenet200':
        net = ResNet18_224x224(num_classes = ID_classes)
    else: 
        net = ResNet18_32x32(num_classes = ID_classes)
    net.load_state_dict(torch.load(ID_model_file))
    
    pw(log, 'Model originally recognized {} classes'
            .format(net.fc.out_features))
    
    # Add one more class to the output
    num_ftrs = net.fc.in_features  # number of input features of the last layer
    # Replace the last layer with a new one with +1 output units
    net.fc = nn.Linear(num_ftrs, net.fc.out_features + 1)  
    
    unlocked = True 
    for param in net.parameters():
        if not param.requires_grad:
            unlocked = False
    pw(log, 'Now ready for recognizing {} classes'
            .format(net.fc.out_features))
    pw(log, 'Model unlocked {}'.format(unlocked))
        
    return net 

''' Builds the augmented traning data and
# the target space containin ID and OOD data '''
def get_augmented_data(sample_rate, **config):
    
    log = config.get('log_filename', "runs/logs/run_tascrec_noconfig.txt")
    ID = config.get('ID', 'cifar10').lower()
    OOD = config.get('OOD', 'cifar100').lower()
    
    mkdirs() 
    
    data = {}
    # get dataloader config
    data_config = get_config(**config)
    data_config.num_workers = 8 
    data_config.parse_refs()
    
    # Load the ID train data
    id_loader_dict = get_dataloader(data_config) 
    pw(log, 'Building {}-org_train'.format(ID))
    with open('data/benchmark_imglist/{}/train_{}.txt'.format(ID, ID)) as f:
        targets = [int(line.split()[1]) for line in f]
    data['org_train'] = ImglistDatasetWrapper(id_loader_dict['train'].dataset, targets)
    
    # Load the ID val data
    pw(log, 'Building {}-org_val'.format(ID))
    with open('data/benchmark_imglist/{}/val_{}.txt'.format(ID, ID)) as f:
        targets = [int(line.split()[1]) for line in f]
    data['org_val'] = ImglistDatasetWrapper(id_loader_dict['val'].dataset, targets)
    
    # Load the ID test data
    pw(log, 'Building {}-org_test'.format(ID))
    with open('data/benchmark_imglist/{}/test_{}.txt'.format(ID, ID)) as f:
        targets = [int(line.split()[1]) for line in f]
    data['org_test'] = ImglistDatasetWrapper(id_loader_dict['test'].dataset, targets)
   
    # create the ood set 
    # The ood labels for the augmented dataset is the ID number of classes
    ID_num_classes = data['org_train'].classes
    print('ID_num_classes: {}'.format(ID_num_classes))
    
    ood_loader_dict = get_ood_dataloader(data_config)
    pw(log, 'Building {}-{}-ood'.format(ID, OOD))
    if 'farood' in ood_loader_dict and OOD in ood_loader_dict['farood']:
        data['ood'] = ImglistDatasetWrapper(ood_loader_dict['farood'][OOD].dataset,
                                            [ID_num_classes] *
                                            len(ood_loader_dict['farood'][OOD].dataset))
    elif 'nearood' in ood_loader_dict and OOD in ood_loader_dict['nearood']:
        data['ood'] = ImglistDatasetWrapper(ood_loader_dict['nearood'][OOD].dataset,
                                            [ID_num_classes] * 
                                            len(ood_loader_dict['nearood'][OOD].dataset))
    else:
        raise Exception(f"'{OOD}' not found in 'farood' or 'nearood' in ood_loader_dict")
        
    
    # Create a random subset of the ood dataset with the same size and class balance
    # as the ID test set
    data['ood'] = torch.utils.data.Subset(data['ood'], 
                    torch.randperm(len(data['ood']))[:len(data['org_test']) // ID_num_classes])
  
    # create the target space
    pw(log, 'Building {}-{}-target.pt'.format(ID,OOD))
    data['target'] = torch.utils.data.ConcatDataset([data['org_test'], data['ood']])
   
    # create the augmented training set and the test set for evaluation 
    
    # Build an augmented training set using test samples
    dataset_indices = list(range(len(data['target'])))
    sample_size = int(sample_rate * len(dataset_indices))
    permuted_indices = torch.randperm(len(dataset_indices)).tolist()
    sample_indices = permuted_indices[:sample_size]
    
    num_uus1 = sum(1 for i in sample_indices if i > len(data['org_test']) - 1)
    num_uus1_str = "{} u.u.s out of {} total examples in the target space sample."
    pw(log, num_uus1_str.format(num_uus1, len(sample_indices))) 
    
    # Remove sampled indices from the dataset indices
    # to create a test set for evaluation
    for idx in sorted(sample_indices, reverse=True):
        dataset_indices.remove(idx)

    pw(log, "Building {}-{}-eval".format(ID, OOD))
    data['eval'] = torch.utils.data.Subset(data['target'], dataset_indices)
    
    pw(log, "Building {}-{}-sample".format(ID, OOD))
    data['sample'] = torch.utils.data.Subset(data['target'], sample_indices)
    
    # Build the augmented dataset 
    pw(log, "Building {}-{}-aug_train".format(ID, OOD))
    aug_train_concat = torch.utils.data.ConcatDataset([data['org_train'], data['sample']])
    # Save the old labels (invisible to TascRec) for developement analysis
    sample_new_targets = [ID_num_classes] * len(data['sample']) 
    data['aug_train'] = HistoricVisionDataset(aug_train_concat,
                                        data['org_train'].targets 
                                        + sample_new_targets)
    
    # Sanity check 
    pw(log, "ID_num_classes: {}".format(ID_num_classes))
    for name in data:
        pw(log, 'Len {}: {}'.format(name, len(data[name])))
    indices = random.sample(range(len(data['sample'])), 30)
    pw(log, "Some examples of the target sample: {}"
        .format([data['sample'][i][1] for i in indices]))
    indices = random.sample(range(len(data['eval'])), 30)
    pw(log, "Some examples of the eval dataset: {}"
        .format([data['eval'][i][1] for i in indices]))
        
    return data 

def train_model(net, dataloaders, num_epochs, 
                early_stop=True, **config):
    
            model = copy.deepcopy(net) 
            
            begin_time = time.time()
            log = config.get('log_filename', "runs/logs/run_tascrec_noconfig.txt")
            opt = config.get('opt', 'adam')
            learning_rate = config.get('learning_rate', 5e-4)
            dataset_sizes = {
                'train': len(dataloaders['train'].dataset),
                'val': len(dataloaders['val'].dataset),
            } 
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                pw(log, 'Using {} GPUs'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            if opt == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            elif opt == 'SGD': 
                optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            else:
                raise ValueError("Invalid optimizer")
    
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.cuda()
            
            # Train the model
            best_model_wts = copy.deepcopy(model.state_dict())
            best_ac = - float("inf")
            best_loss = float("inf")
            best_model_epoch = 0
            
            for epoch in range(num_epochs):
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  
                    else:
                        model.eval()
                    running_loss = 0.0
                    running_corrects = 0.0
                    
                    for batch in dataloaders[phase]:
                        inputs = batch[0].cuda()
                        labels = batch[1].cuda()
                        
                        optimizer.zero_grad()

                        # forward
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        running_loss += loss.item() * inputs.size(0) 
                        running_corrects += torch.sum(preds == labels.data)
                    # Compute the epoch loss
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase] 
                    if phase == 'val':
                        val_loss= epoch_loss
                    
                    epoch_str = '{}-{} | Epoch {:03d}/{:03d} | Time: {:5d}s | {:5} loss: {:.4f} Acc: {:.4f} | {} predictions'\
                        .format(config['ID'], config['OOD'], epoch+1, num_epochs, int(time.time() - begin_time),
                                phase, epoch_loss, epoch_acc, dataset_sizes[phase])
                    pw(log, epoch_str)
                    # Save the model if it has the best accuracy
                    if early_stop and phase == 'val' and epoch_loss < best_loss:
                        #print('Found a better model', epoch_acc, best_ac)
                        best_loss = epoch_loss
                        best_model_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict()) 
                
                if opt == 'adam':
                    scheduler.step(val_loss)  
                elif opt == 'SGD':
                    scheduler.step() 
                    
            pw(log, 'Best model epoch: {}'.format(best_model_epoch)) 
            model.load_state_dict(best_model_wts)
            return model

# Assuming 'dataset' is HistoricVisionDataset. 'net' is the DNN model class
def cross_reclassification(net, dataset, rebuild=0, num_epochs= 2, n_splits=2,
                           **config):
    
     
    log = config.get('log_filename', 
                     "runs/logs/run_tascrec_noconfig.txt")
    validation_split= config.get('validation_split', 0.15)
    
    assert isinstance(dataset, HistoricVisionDataset), \
        "dataset is not an instance of HistoricVisionDataset" 
        
    ID = config.get('ID', 'cifar10')
    OOD = config.get('OOD', 'cifar100')
    mkdirs()
    
    file_path = ('runs/data_cache/{}{}_uus_class_set-split_{}_epochs_{}.pt'
        .format(ID, OOD, n_splits, num_epochs))
    stats_path = file_path + '_stats.json'
    if os.path.exists(file_path) and not rebuild:
        uus_class_set = torch.load(file_path)
        pw(log, "Loaded mined u.u.s dataset from {}".format(file_path))
    else:
        pw(log, "Building dataset {}".format(file_path))
        OOD_class = max(dataset.targets)
        
        oodc_str = "OOD_class for CR: {}".format(OOD_class)
        pw(log, 'Cross-Reclassification')
        pw(log, oodc_str)
         
        stats= {} 
        kfold = KFold(n_splits=n_splits, shuffle=True)
    
        uus_input = []
        dev_cum_uus_old_labels = []
        dev_cum_actual_uus_count = 0 
        dev_cum_reclassified = 0
        since = time.time() 
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # Re-initialize the model 
            model = copy.deepcopy(net)
            
            train_dataset_fold = torch.utils.data.Subset(dataset, train_ids)
            
            train_size = int((1 - validation_split) * len(train_dataset_fold))
            val_size = len(train_dataset_fold) - train_size
            train_fold, val_fold = torch.utils.data.random_split(train_dataset_fold, 
                                                                 [train_size, val_size])
            
            # Reclassify only the samples of the added dummy class
            reclass_ids = [id for id in test_ids if dataset[id][1] == OOD_class]
            reclass_dataset_fold = torch.utils.data.Subset(dataset, reclass_ids)
        
            pw(log, '_'*30)
            pw(log, f"CR FOLD {fold}")
            pw(log, "Training fold size: {}".format(len(train_dataset_fold)))
            pw(log, "Train size: {}".format(len(train_fold)))
            pw(log, "Val size: {}".format(len(val_fold)))
            pw(log, "Reclass fold size: {}".format(len(reclass_dataset_fold)))
           
            workers = config.get('NUM_WORKERS', 0)
            # Define data loaders for training and testing data in this fold
            trainloader = DataLoader(train_fold, batch_size=256, shuffle=True, \
                pin_memory=True, num_workers=workers, drop_last=False)
            validationloader = DataLoader(val_fold, batch_size=256, shuffle=True, \
                pin_memory=True, num_workers=workers, drop_last=False)
            reclassloader = DataLoader(reclass_dataset_fold, batch_size=256, shuffle=True, \
                pin_memory=True, num_workers=workers, drop_last=False)
            
            dataloaders = {'train': trainloader, 'val': validationloader}

            trained_model = train_model(model, dataloaders, num_epochs, **config) 
            
            # mine the u.u.s
            trained_model.eval()
            with torch.no_grad():
                for inputs, labels, old_labels in reclassloader:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    old_labels = old_labels.cuda()
                    
                    outputs = trained_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    uus_input.append(inputs[predicted == OOD_class])
                    
                    # collect data for analysis
                    # a list of the old uu labels
                    dev_cum_uus_old_labels += old_labels[predicted == OOD_class].tolist()
                    # accumulated the number of examples submitted to cross reclassification
                    dev_cum_reclassified += len(predicted)
                # count uus that we actually found
                dev_cum_actual_uus_count = sum([1 for e in dev_cum_uus_old_labels if e == OOD_class])
                pw(log, "So far, we mined {} u.u.s".format(len(dev_cum_uus_old_labels)))
                pw(log, "of which {} are actual u.u.s from a total of {} predictions".format(dev_cum_actual_uus_count,  dev_cum_reclassified))
                pw(log, "u.u.s' old labels: {}".format(dev_cum_uus_old_labels))
                    
        time_elapsed = time.time() - since
        stats['org_labels'] = dev_cum_uus_old_labels
        stats['len'] = len(dev_cum_uus_old_labels)
        stats['actual'] = dev_cum_actual_uus_count
        stats['epochs'] = num_epochs
        stats['folds'] = n_splits
        stats['log'] = log
        stats['time(s)'] = time_elapsed 
        with open(stats_path, 'w') as f:
            json.dump(stats, f) 
        pw(log, 'CR completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))      
            
        # make uus_class (features_tensor, labels_tensor) where all labels are OOD_class
        uus_input_tensor = torch.cat(uus_input, dim=0).cpu()
        pw(log, 'Total u.u.s examples in tensor: {}'.format(len(uus_input_tensor)))

        labels_list = [OOD_class] * len(uus_input_tensor)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long).cpu()
        
        uus_class_set = torch.utils.data.TensorDataset(uus_input_tensor, labels_tensor)

        torch.save(uus_class_set, file_path)
    return uus_class_set

# Rectify the model using the mined u.u.s
def rectify_model(net, org_train_set, uus_mined_set,
                  rebuild=False, num_epochs=3, **config):
    
    final_model = copy.deepcopy(net)
    # we make the targets a tensor here
    # recopy the data to avoid doing it twice when repeating rectification
    org_train_set = copy.deepcopy(org_train_set) 
    
    log = config.get('log_filename', 
                     "runs/logs/run_tascrec_noconfig.txt")
    workers = config.get('NUM_WORKERS', 0)
    validation_split = config.get('validation_split', 0.1)
    OOD = config.get('OOD', 'cifar10-cifar100')
    mkdirs() 
    rectified_model_filename = ("runs/models/rectified_{}_epochs_{}.pth"
                                .format(OOD, num_epochs))
    
        
    pw(log, "Model rectification")
        
    if os.path.exists(rectified_model_filename) and not rebuild:
        pw(log, "Rectified model already exists.")
        
        '''check if the state dictionary was saved with nn.DataParallel
        and load it into a non-DataParallel model 
        by modifying the keys in the state dictionary to remove 
        the 'module.' prefix if it exists.'''
        state_dict = torch.load(rectified_model_filename)

        # Create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        final_model.load_state_dict(new_state_dict)
        
    else:
        pw(log, "Model rectification started")
       
        # align the format of the datasets to concat them
        org_train_set.targets = torch.tensor(org_train_set.targets, dtype=torch.long)
        dataset = torch.utils.data.ConcatDataset([org_train_set, uus_mined_set])
        dataset_targets = [target for _, target in dataset]
        
        class_counts = torch.bincount(org_train_set.targets)
        uus_mined_set_length = torch.tensor([len(uus_mined_set)], dtype=torch.float32)
        class_counts = torch.cat((class_counts, uus_mined_set_length))
        class_weights = 1.0 / class_counts.float()
        
        pw(log, "Length of uus_mined_set: {}".format(uus_mined_set_length))
        pw(log, "Class cardinality: {}".format(class_counts))
        pw(log, "Class weights: {}".format(class_weights))
        
        # Split the dataset into training and validation sets
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        len_str = 'len dataset {}, train size {}, val size {}'\
            .format(len(dataset), train_size, val_size)
        pw(log, len_str)
        
        # Create a list of class labels for the training set
        train_targets = [dataset_targets[i] for i in train_dataset.indices] 
        
        # Create a WeightedRandomSampler for the training set
        weights = [class_weights[i] for i in train_targets]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_dataset)) 
        balanced = config.get('balanced', True) 
        # Create data loaders with the WeightedRandomSampler
        if balanced:
            train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler, num_workers=workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=workers, pin_memory=True)
        validation_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
       
        final_model = train_model(final_model, {'train': train_loader, 'val': validation_loader},
                            num_epochs, **config)
        
        # Save the rectified model to a file
        #torch.save(final_model.state_dict(), rectified_model_filename)
    
    return final_model.state_dict()

def evaluate_model(net, weights, test_dataset, **config):
   
    model = copy.deepcopy(net)
    
    log = config.get('log_filename', 
                     "runs/logs/run_tascrec_noconfig.txt")
    workers = config.get('NUM_WORKERS', 0)
    mkdirs()

    '''Check if the state dictionary was saved with nn.DataParallel
    and load it into a non-DataParallel model by modifying 
    the keys in the state dictionary to remove the 'module.' 
    prefix if it exists.
    Create new OrderedDict without "module."
    '''
    new_state_dict = OrderedDict()

    for k, v in weights.items():
        if k.startswith('module.'):
            name = k[7:]  # remove "module."
        else:
            name = k  # use original key
        new_state_dict[name] = v

    # Load it into the model
    model.load_state_dict(new_state_dict) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        pw(log, "Rectify using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.eval()
    
    # extrac the max label od test_dataset
    OOD_class = max(label for _, label in test_dataset)
    
    #print OOD_class to log_file
    ood_str = "OOD_class for evaluation: {}".format(OOD_class)
    pw(log, ood_str)
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
    
    y_true = []
    y_ood_true = []
    y_pred = []
    y_ood_pred = []
    y_ood_score = [] 
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            score = softmax(outputs.data, dim = 1)[:,OOD_class] 
            
            # Convert labels to binary: OOD_class is 1, other classes are 0
            ood_labels_binary = (labels == OOD_class).float()
            ood_predicted_binary = (predicted == OOD_class).float()
            
            y_ood_true.extend(ood_labels_binary.tolist())
            y_ood_pred.extend(ood_predicted_binary.tolist())
            y_ood_score.extend(score.tolist())
           
            # Filter labels and predicted 
            # to computer accuracy using non_ood_indices
            non_ood_indices = (labels != OOD_class)
            non_ood_labels = labels[non_ood_indices]
            non_ood_predicted = predicted[non_ood_indices]

            y_true.extend(non_ood_labels.tolist())
            y_pred.extend(non_ood_predicted.tolist()) 
            
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_ood_true, y_ood_score)
    precision, recall, _ = precision_recall_curve(y_ood_true,y_ood_score)
    aupr = auc(recall,precision)
    
    # Compute FPR@95: first find the threshold where the recall is 95%
    fpr, tpr, thresholds = roc_curve(y_ood_true, y_ood_score)
    # Find the threshold where TPR is closest to 95% but not less.
    target_tpr_index = np.where(tpr >= 0.95)[0][0]
    fpr_95 = fpr[target_tpr_index]
 
    result_header = "Acc\tAUROC\tAUPR\tFPR@95\n" 
    result_str = "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(accuracy, auroc, aupr, fpr_95)
        
    pw(log, result_header + result_str)
    
    return accuracy, auroc, aupr, fpr_95
