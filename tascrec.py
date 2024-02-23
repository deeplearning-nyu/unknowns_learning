import sys
sys.path.append('/home/abrahao/data/bd58/uus/OpenOOD')


# necessary imports
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import gdown
import zipfile
import os
import time

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network


NUM_WORKERS = 35 
learning_rate = 5e-4

def get_resnet18():
    # download our pre-trained CIFAR-10 classifier
    model_file= '/home/abrahao/data/bd58/uus/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'

    # Check if the zip file already exists
    if not os.path.exists(model_file):
        # download our pre-trained CIFAR-10 classifier
        url= 'https://drive.google.com/uc?id=1byGeYxM_PlLjT72wZsMQvP6popJeWBgt'
        zip_file_path= '/home/abrahao/data/bd58/uus/results/cifar10_res18_v1.5.zip'
        gdown.download(url, zip_file_path, quiet=False)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_directory)
        print("Model download and extraction complete.")
    else: 
        print("Model already downloaded.")

    # load the model
    net= ResNet18_32x32(num_classes=10)
    net.load_state_dict(
        torch.load('results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
    )

    # Add one more class to the output
    num_ftrs = net.fc.in_features  # Get the number of input features of the last layer
    net.fc = nn.Linear(num_ftrs, 11)  # Replace the last layer with a new one with 11 output units

    return net 




# get the number and IDs of GPUs
device_count = torch.cuda.device_count()
device_ids = list(range(device_count))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("run.txt","a") as f:
    f.write("This RUN uses {} GPU\n".format(device_count))
print("This RUN uses {} GPU\n".format(device_count))

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print('loading data')
# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = get_resnet18()
model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

criterion = nn.CrossEntropyLoss()
optimizer = optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print('training')
# Train the model
num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print the loss every 1000 mini-batches
        running_loss += loss.item()
        if i % 100 == 0:
            print('[Epoch {}/{} - Batch {}/{}] Loss: {:.4f}'.format(epoch+1, num_epochs, i, len(train_loader), running_loss/1000))
            running_loss = 0.0

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy: {}'.format(100 * correct / total))