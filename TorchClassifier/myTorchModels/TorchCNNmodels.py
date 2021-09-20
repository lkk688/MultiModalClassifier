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
    elif name=='mlpmodel1':
        return create_mlpmodel1(numclasses, img_shape)
    elif name=='lenet':
        return create_lenet(numclasses, img_shape)
    elif name=='alexnet':
        return create_AlexNet(numclasses, img_shape)
    elif name=='vggmodel1':
        return create_vggmodel1(numclasses, img_shape)
    elif name=='vggcustom':
        return create_vggcustommodel(numclasses, img_shape)
    elif name=='resnetmodel1':
        return create_resnetmodel1(numclasses, img_shape)

def create_vggmodel1(numclasses, img_shape):
    # Load the pretrained model from pytorch
    vgg16 = models.vgg16(pretrained=True)

    # print out the model structure
    print(vgg16)
    print(vgg16.classifier[6].in_features) #4096
    print(vgg16.classifier[6].out_features) #1000

    #Freeze training for all "features" layers
    for param in vgg16.features.parameters():
        param.requires_grad = False
    
    #Final Classifier Layer, Once you have the pre-trained feature extractor, you just need to modify and/or add to the final, fully-connected classifier layers. Replace the last layer in the vgg classifier group of layers.
    n_inputs = vgg16.classifier[6].in_features

    # add last linear layer (n_inputs -> 5 flower classes)
    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, numclasses)

    vgg16.classifier[6] = last_layer
    
class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        
        self.features = features
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

def create_vggcustommodel(numclasses, img_shape):
    #Each item in the list is either 'M', which denotes a max pooling layer, or an integer, which denotes a convolutional layer with that many filters.
    vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 
                    512, 'M']

    vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
                    512, 512, 512, 512, 'M']

    vgg11_layers = get_vgg_layers(vgg11_config, batch_norm = True)
    print(vgg11_layers)

    OUTPUT_DIM = numclasses
    model = VGG(vgg11_layers, OUTPUT_DIM)

    print(model)

    #pre-trained VGG11 with batch normalization, 140MB
    #import torchvision.models as models
    pretrained_model = models.vgg11_bn(pretrained = True)

    print(pretrained_model)
    #the pre-trained model loaded is exactly the same as the one we have defined with one exception - the output of the final linear layer.
    #All of torchvision's pre-trained models are trained as image classification models on the ImageNet dataset. 
    #A dataset of 224x224 color images with 1000 classes, therefore the final layer will have a 1000 dimensional output.
    #We can get the last layer specifically by indexing into the classifier layer of the pre-trained model.
    print("VGG last layer size:", pretrained_model.classifier[-1])
    #We'll define a new final linear layer which has to have an input size equal to that of the layer we are replacing - as it's input will be the 4096 dimensional output from the previous linear layer in the classifier.
    IN_FEATURES = pretrained_model.classifier[-1].in_features 
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) #final_fc will be initialized randomly. It is the only part of our model with its parameters not pre-trained.
    pretrained_model.classifier[-1] = final_fc
    print(pretrained_model.classifier)

    #We can load the parameters of the pretrained_model into our model by loading the parameters (state_dict) from the pretrained_model into our model
    #This is only possible as our model has the exact same layers (order and shape) as the pretrained_model with the final linear layer replaced
    model.load_state_dict(pretrained_model.state_dict())
    num_trainparameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_trainparameters} trainable parameters')

    #Instead of training all of the parameters we have loaded from a pre-trained model, we could instead only learn some of them and leave some "frozen" at their pre-trained values. 
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    #We could also freeze the classifier layer, however we always want to train the last layer
    for parameter in model.classifier[:-1].parameters():
        parameter.requires_grad = False

def get_vgg_layers(config, batch_norm):

    #Batch normalization: BN is a layer with learnable parameters - two per filter denoted  γ  and  β
    #We first calculate the mean and variance across each channel dimension of the batch
    #Then normalize the batch by subtracting the channel means and dividing by the channel stds
    #We then scale and shift each channel of this normalized batch of inputs,  x^i  using  γ  and  β
    #there is a better mean and std for our task instead of zero and one. If this is the case then our model can learn this whilst training as  γ  and  β  are learnable parameters

    layers = []
    in_channels = 3
    
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c
            
    return nn.Sequential(*layers)

class CNNNet1(nn.Module): #32*32 image input
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


class MLP(nn.Module): #for MNIST dataset
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #three linear layers
        #take the input batch of images and flatten them so they can be passed into the linear layers
        self.input_fc = nn.Linear(input_dim, 250) #hidden dimensions of 250 elements
        self.hidden_fc = nn.Linear(250, 100) #hidden dimensions of 100 elements
        self.output_fc = nn.Linear(100, output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0] #x = [batch size, height, width]

        x = x.view(batch_size, -1)
        #x = [batch size, height * width]
        
        h_1 = F.relu(self.input_fc(x))
        #h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        #h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)
        #y_pred = [batch size, output dim]
        
        return y_pred, h_2

def create_mlpmodel1(numclasses, img_shape):
    #for MNIST dataset
    INPUT_DIM = img_shape[1]*img_shape[2]#28 * 28
    OUTPUT_DIM = numclasses
    model = MLP(INPUT_DIM, OUTPUT_DIM)
    print(model)

    for p in model.parameters():
        if p.requires_grad:
            print("trainable parameters:", p.numel()) #PyTorch torch.numel() method returns the total number of elements in the input tensor.
    return model


class LeNet(nn.Module):#for 28*28 MNIST dataset
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 6, 
                               kernel_size = 5)
        
        self.conv2 = nn.Conv2d(in_channels = 6, 
                               out_channels = 16, 
                               kernel_size = 5)
        
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):

        #x = [batch size, 1, 28, 28]
        
        x = self.conv1(x)
        
        #x = [batch size, 6, 24, 24]
        
        x = F.max_pool2d(x, kernel_size = 2)
        
        #x = [batch size, 6, 12, 12]
        
        x = F.relu(x)
        
        x = self.conv2(x)
        
        #x = [batch size, 16, 8, 8]
        
        x = F.max_pool2d(x, kernel_size = 2)
        
        #x = [batch size, 16, 4, 4]
        
        x = F.relu(x)
        
        x = x.view(x.shape[0], -1)
        
        #x = [batch size, 16*4*4 = 256]
        
        h = x
        
        x = self.fc_1(x)
        
        #x = [batch size, 120]
        
        x = F.relu(x)

        x = self.fc_2(x)
        
        #x = batch size, 84]
        
        x = F.relu(x)

        x = self.fc_3(x)

        #x = [batch size, output dim]
        
        return x, h

def create_lenet(numclasses, img_shape):
    #for MNIST dataset
    OUTPUT_DIM = numclasses
    model = LeNet(OUTPUT_DIM)
    print(model)

    num_trainparameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_trainparameters} trainable parameters')
    return model


class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2), #kernel_size
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 192, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 384, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

def create_AlexNet(numclasses, img_shape):
    #for MNIST dataset
    #INPUT_DIM = img_shape[1]*img_shape[2]#28 * 28
    OUTPUT_DIM = numclasses
    model = AlexNet(OUTPUT_DIM)
    print(model)

    for p in model.parameters():
        if p.requires_grad:
            print("trainable parameters:", p.numel()) #PyTorch torch.numel() method returns the total number of elements in the input tensor.
    num_trainparameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_trainparameters} trainable parameters')
    return model


def create_resnetmodel1(numclasses, img_shape):
    model_ft = models.resnet18(pretrained=True) #Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/lkk/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
    num_ftrs = model_ft.fc.in_features #512
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, numclasses)
    return model_ft
