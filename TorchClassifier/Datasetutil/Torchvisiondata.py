#This code should be run in the head node with internet access to download the data
import torch
import torchvision
import torchvision.datasets as datasets
import os

print(torch.__version__)
cifar_trainset = datasets.CIFAR10(root='/data/cmpe249-fa22/torchvisiondata/', train=True, download=True, transform=None)
print(cifar_trainset)
#Extracting /data/cmpe249-fa22/torchvisiondata/cifar-10-python.tar.gz to /data/cmpe249-fa22/torchvisiondata/
cifar_testset = datasets.CIFAR10(root='/data/cmpe249-fa22/torchvisiondata/', train=False, download=True, transform=None)
print(cifar_testset)
#Extracting ./data/cifar-10-python.tar.gz to ./data

mnist_testset = datasets.MNIST(root='/data/cmpe249-fa22/torchvisiondata/', train=False, download=True, transform=None)


#Download model
os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/' #setting the environment variable
resnet18 = torchvision.models.resnet18(pretrained=True)
print(resnet18)
resnet50 = torchvision.models.resnet50(pretrained=True)
print(resnet50)
#Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /data/cmpe249-fa22/torchhome/hub/checkpoints/resnet50-19c8e357.pth