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

mytorchvisiondata='/data/cmpe249-fa22/torchvisiondata/'
training_data = datasets.FashionMNIST(
    root=mytorchvisiondata,
    train=True,
    download=True,
    transform=None,
)
test_data = datasets.FashionMNIST(
    root=mytorchvisiondata,
    train=False,
    download=True,
    transform=None,
)
