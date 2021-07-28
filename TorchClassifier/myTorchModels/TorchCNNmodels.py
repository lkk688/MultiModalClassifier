import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

def createTorchCNNmodel(name, numclasses, img_shape):
    if name=='cnnmodel1':
        return create_cnnmodel1(numclasses, img_shape)
    elif name=='resnetmodel1':
        return create_resnetmodel1(numclasses, img_shape)

class CNNNet1(nn.Module):
    def __init__(self, numclasses):
        super(CNNNet1, self).__init__()

        #CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') from https://pytorch.org/docs/stable/nn.html
        #We can compute the spatial size of the output volume as a function of the input volume size (W), the kernel/filter size (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. The correct formula for calculating how many neurons define the output_W is given by (W−F+2P)/S+1.

        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #Applies a 2D convolution over an input signal composed of several input planes.
        #16 filters: output size after Conv = (32-3+2*1)/1+1=32, output size: 32*32*16

        # convolutional layer (sees 16x16x16 tensor) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        #input 16*16*16 (pooling from 32*32*16), output (16-3+2)+1=16, output=16*16*32

        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # input 8*8*32 (pooling from 16*16*32), output (8-3+2)+1=8, output=8*8*64

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, numclasses)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x))) # output size: 32*32*16, pool=16*16*16
        x = self.pool(F.relu(self.conv2(x))) # output 16*16*16, pool=8*8*16
        x = self.pool(F.relu(self.conv3(x))) # 8*8*64, pool=4*4*64

        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))# 64 * 4 * 4 -> 500
        # add dropout layer 
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x) # 500 -> 10
        return x

def create_cnnmodel1(numclasses, img_shape):
    
    # define the CNN architecture
    model = CNNNet1(numclasses)
    print(model)
    return model


def create_resnetmodel1(numclasses, img_shape):
    model_ft = models.resnet18(pretrained=True) #Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/lkk/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
    num_ftrs = model_ft.fc.in_features #512
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, numclasses)
    return model_ft
