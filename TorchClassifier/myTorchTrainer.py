from __future__ import print_function, division
import configargparse #pip install configargparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
#import matplotlib.pyplot as plt
import time
import os
import copy
import random

import PIL
import PIL.Image

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)

from TorchClassifier.Datasetutil.Visutil import imshow, vistestresult, matplotlib_imshow
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel
from TorchClassifier.myTorchModels.TorchOptim import gettorchoptim
from TorchClassifier.myTorchModels.TorchLearningratescheduler import setupLearningratescheduler
# from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset #loadtfds, loadkerasdataset, loadimagefolderdataset
# from TFClassifier.myTFmodels.CNNsimplemodels import createCNNsimplemodel
# from TFClassifier.Datasetutil.Visutil import plot25images, plot9imagesfromtfdataset, plot_history
# from TFClassifier.myTFmodels.optimizer_factory import build_learning_rate, setupTensorboardWriterforLR

model = None 
device = None
# import logger

# Test CIFAR10:
#python myTorchTrainer.py --data_name 'CIFAR10' --data_type 'torchvisiondataset' --data_path r"E:\Dataset" --model_name 'cnnmodel1' --learningratename 'ConstantLR' --optimizer 'SGD'

#python myTorchTrainer.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' --data_path r"E:\Dataset\ImageNet\tiny-imagenet-200" --model_name 'resnetmodel1' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'SGD'


parser = configargparse.ArgParser(description='myTorchClassify')
parser.add_argument('--data_name', type=str, default='tiny-imagenet-200',
                    help='data name: tiny-imagenet-200, hymenoptera_data, CIFAR10, MNIST, flower_photos')
parser.add_argument('--data_type', default='trainonly', choices=['trainvalfolder', 'traintestfolder', 'torchvisiondataset'],
                    help='the type of data') 
parser.add_argument('--data_path', type=str, default=r"E:\Dataset\ImageNet\tiny-imagenet-200",
                    help='path to get data') #/Developer/MyRepo/ImageClassificationData; r"E:\Dataset\ImageNet\tiny-imagenet-200"
parser.add_argument('--img_height', type=int, default=224,
                    help='resize to img height, 224')
parser.add_argument('--img_width', type=int, default=224,
                    help='resize to img width, 224')
parser.add_argument('--save_path', type=str, default='./outputs/',
                    help='path to save the model')
# network
parser.add_argument('--model_name', default='resnetmodel1', choices=['mlpmodel1', 'lenet', 'alexnet', 'resnetmodel1', 'customresnet', 'vggmodel1', 'vggcustom', 'cnnmodel1'],
                    help='the network')
parser.add_argument('--arch', default='Pytorch', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Pytorch.')
parser.add_argument('--learningratename', default='StepLR', choices=['StepLR', 'ConstantLR' 'ExponentialLR', 'MultiStepLR', 'OneCycleLR'],
                    help='learning rate name')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'adamresnetcustomrate'],
                    help='select the optimizer')
parser.add_argument('--batchsize', type=int, default=128,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=40,
                    help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
# parser.add_argument('--TPU', type=bool, default=False,
#                     help='use TPU')
# parser.add_argument('--MIXED_PRECISION', type=bool, default=False,
#                     help='use MIXED_PRECISION')
parser.add_argument('--TAG', default='0326',
                    help='setup the experimental TAG to differentiate different running results')
parser.add_argument('--reproducible', type=bool, default=False,
                    help='get reproducible results we can set the random seed for Python, Numpy and PyTorch')


args = parser.parse_args()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, 
                tensorboard_writer=None, profile=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss = [0.0 for i in range(num_epochs)]
    val_loss = [0.0 for i in range(num_epochs)]
    train_acc = [0.0 for i in range(num_epochs)]
    val_acc = [0.0 for i in range(num_epochs)]

    if profile is not None:
        profile.start()

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
                    if type(outputs) is tuple: #model may output multiple tensors as tuple
                        outputs, _ = outputs
                    # convert output probabilities to predicted class
                    _, preds = torch.max(outputs, 1)#outputs size [32, 10]

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
                    
                if profile is not None:
                    profile.step()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_loss[epoch] = epoch_loss
                train_acc[epoch] = epoch_acc
            else:
                val_loss[epoch] = epoch_loss
                val_acc[epoch] = epoch_acc
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        #finish one epoch train and val
        print()
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalars('Training vs. Validation Loss',
                {'Training': train_loss[epoch], 'Validation': val_loss[epoch]},
                epoch)
            tensorboard_writer.add_scalars('Training vs. Validation Accuracy',
                {'Training': train_acc[epoch], 'Validation': val_acc[epoch]},
                epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if tensorboard_writer is not None:
        tensorboard_writer.flush()
    if profile is not None:
        profile.stop()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
            if type(outputs) is tuple: #model may output multiple tensors as tuple
                outputs, _ = outputs
            # convert output probabilities to predicted class
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

def visualize_result(model, dataloaders, classes, key='val'):
    images, labels = next(iter(dataloaders['val']))
    # move model inputs to cuda, if GPU available
    images = images.to(device)
    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    #preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    preds = np.squeeze(preds_tensor.cpu().numpy())
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        imshow(images.cpu()[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))


def test_model(model, dataloaders, class_names, criterion, batch_size, key='test'):
    numclasses = len(class_names)
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(numclasses))
    class_total = list(0. for i in range(numclasses))

    model.eval()

    if key in dataloaders.keys():
        test_loader=dataloaders[key]
    else:
        print("test dataset not available")
        return

    # iterate over test data
    bathindex = 0
    for data, target in test_loader:
        bathindex = bathindex +1
        # move tensors to GPU if CUDA is available
        # if train_on_gpu:
        #     data, target = data.cuda(), target.cuda()
        data = data.to(device)
        target = target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(data)
        if type(outputs) is tuple: #model may output multiple tensors as tuple
            outputs, _ = outputs
        # calculate the batch loss
        loss = criterion(outputs, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(outputs, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        train_on_gpu = torch.cuda.is_available()
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

        # calculate test accuracy for each object class
        for i in range(batch_size):
            if i<len(target.data):#the actual batch size of the last batch is smaller than the batch_size
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

def main():
    print("Torch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if args.reproducible==True:
        #https://pytorch.org/docs/stable/notes/randomness.html
        SEED = 1234
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    #TAG="0727"
    args.save_path=args.save_path+args.data_name+'_'+args.model_name+'_'+args.TAG
    print("Output path:", args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # torch.utils.tensorboard.SummaryWriter is imported above
    #tensorboard --logdir= ; open a browser tab to http://localhost:6006/
    tensorboard_writer = SummaryWriter(args.save_path)

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
    #Visualize dataset
    test_loader=dataloaders['train']
    # obtain one batch of test images
    images, labels = next(iter(test_loader))
    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=False)
    tensorboard_writer.add_image('Image-grid',img_grid)
    tensorboard_writer.flush()

    numclasses =len(class_names)
    model_ft = createTorchCNNmodel(args.model_name, numclasses, img_shape)

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    tensorboard_writer.add_graph(model_ft, images)
    tensorboard_writer.flush()

    criterion = nn.CrossEntropyLoss()

    model_ft = model_ft.to(device)
    criterion = criterion.to(device)

    # Observe that all parameters are being optimized, 
    optimizer_ft=gettorchoptim(args.optimizer, model_ft, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay) #'Adam'
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters())

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    STEPS_PER_EPOCH = len(dataloaders['train'])
    lr_scheduler = setupLearningratescheduler(args.learningratename, optimizer_ft, args.epochs, STEPS_PER_EPOCH)

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.save_path),
        record_shapes=True,
        with_stack=True)
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, lr_scheduler,
                       num_epochs=args.epochs, tensorboard_writer=tensorboard_writer, profile=prof)

    #save torch model
    modelsavepath = os.path.join(args.save_path, 'model_best.pt')
    torch.save(model_ft.state_dict(), modelsavepath)

    test_model(model_ft, dataloaders, class_names, criterion, args.batchsize, key='val')
    
    #visualize_model(model_ft, dataloaders, class_names, num_images=6)
    #visualize_result(model_ft, dataloaders, class_names, key='val')
    


if __name__ == '__main__':
    main()
