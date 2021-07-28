from __future__ import print_function, division
import configargparse #pip install configargparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import PIL
import PIL.Image

print(torch.__version__)

from TorchClassifier.Datasetutil.Visutil import imshow, vistestresult
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel
# from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset #loadtfds, loadkerasdataset, loadimagefolderdataset
# from TFClassifier.myTFmodels.CNNsimplemodels import createCNNsimplemodel
# from TFClassifier.Datasetutil.Visutil import plot25images, plot9imagesfromtfdataset, plot_history
# from TFClassifier.myTFmodels.optimizer_factory import build_learning_rate, setupTensorboardWriterforLR

model = None 
device = None
# import logger

parser = configargparse.ArgParser(description='myTorchClassify')
parser.add_argument('--data_name', type=str, default='CIFAR10',
                    help='data name: hymenoptera_data, CIFAR10')
parser.add_argument('--data_type', default='torchvisiondataset', choices=['trainvalfolder', 'torchvisiondataset'],
                    help='the type of data') 
parser.add_argument('--data_path', type=str, default='/Developer/MyRepo/ImageClassificationData',
                    help='path to get data') 
parser.add_argument('--img_height', type=int, default=224,
                    help='resize to img height')
parser.add_argument('--img_width', type=int, default=224,
                    help='resize to img width')
parser.add_argument('--save_path', type=str, default='./outputs/',
                    help='path to save the model')
# network
parser.add_argument('--model_name', default='cnnmodel1', choices=['resnetmodel1', 'cnnmodel1'],
                    help='the network')
parser.add_argument('--arch', default='Pytorch', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Pytorch.')
parser.add_argument('--learningratename', default='warmupexpdecay', choices=['fixedstep', 'fixed', 'warmupexpdecay'],
                    help='path to save the model')
parser.add_argument('--batchsize', type=int, default=32,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=15,
                    help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use TPU')
parser.add_argument('--MIXED_PRECISION', type=bool, default=False,
                    help='use MIXED_PRECISION')


args = parser.parse_args()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients, clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #forward pass: compute predicted outputs by passing inputs to the model
                    outputs = model(inputs) #shape 4,2; 32,10
                    _, preds = torch.max(outputs, 1)

                    # calculate the batch loss
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # perform a single optimization step (parameter update)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)#batch size
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloaders, class_names, criterion, batch_size):
    numclasses = len(class_names)
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(numclasses))
    class_total = list(0. for i in range(numclasses))

    model.eval()

    if 'test' in dataloaders.keys():
        test_loader=dataloaders['test']
    else:
        print("test dataset not available")
        return

    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        # if train_on_gpu:
        #     data, target = data.cuda(), target.cuda()
        data = data.to(device)
        target = target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        train_on_gpu = torch.cuda.is_available()
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(numclasses):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                class_names[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_names[i]))
    
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():
    print("Torch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    TAG="0727"
    args.save_path=args.save_path+args.data_name+'_'+args.model_name+'_'+TAG
    print("Output path:", args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.GPU:
        num_gpu = torch.cuda.device_count()
        print("Num GPUs:", num_gpu)
        # Which GPU Is The Current GPU?
        print(torch.cuda.current_device())

        # Get the name of the current GPU
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # Is PyTorch using a GPU?
        print(torch.cuda.is_available())
        global device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    else:
        print("No GPU and TPU enabled")
    
    #Load dataset
    dataloaders, dataset_sizes, class_names, img_shape = loadTorchdataset(args.data_name,args.data_type, args.data_path, args.img_height, args.img_width, args.batchsize)

    numclasses =len(class_names)
    model_ft = createTorchCNNmodel(args.model_name, numclasses, img_shape)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args.epochs)

    #save torch model
    modelsavepath = os.path.join(args.save_path, 'model_best.pt')
    torch.save(model_ft.state_dict(), modelsavepath)
    
    visualize_model(model_ft, dataloaders, class_names, num_images=6)

    test_model(model_ft, dataloaders, class_names, criterion, args.batchsize)

    if 'test' in dict.keys():
        test_loader=dataloaders['test']
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images.numpy()
        images = images.to(device)

        # get sample outputs
        output = model_ft(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())

        vistestresult(images, labels, preds, class_names)

if __name__ == '__main__':
    main()
