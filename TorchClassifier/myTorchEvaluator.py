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
parser.add_argument('--data_name', type=str, default='MNIST',
                    help='data name: hymenoptera_data, CIFAR10, flower_photos')
parser.add_argument('--data_type', default='torchvisiondataset', choices=['trainvalfolder', 'traintestfolder', 'torchvisiondataset'],
                    help='the type of data') 
parser.add_argument('--data_path', type=str, default='./../ImageClassificationData',
                    help='path to get data') #/Developer/MyRepo/ImageClassificationData
parser.add_argument('--img_height', type=int, default=28,
                    help='resize to img height, 224')
parser.add_argument('--img_width', type=int, default=28,
                    help='resize to img width, 224')
parser.add_argument('--save_path', type=str, default='./outputs/',
                    help='path to save the model')
# network
parser.add_argument('--model_name', default='lenet', choices=['mlpmodel1', 'lenet', 'resnetmodel1', 'vggmodel1', 'cnnmodel1'],
                    help='the network')
parser.add_argument('--arch', default='Pytorch', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Pytorch.')
parser.add_argument('--learningratename', default='warmupexpdecay', choices=['fixedstep', 'fixed', 'warmupexpdecay'],
                    help='learning rate name')
parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adam'],
                    help='select the optimizer')
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
parser.add_argument('--TAG', default='0915',
                    help='setup the experimental TAG to differentiate different running results')
parser.add_argument('--reproducible', type=bool, default=False,
                    help='get reproducible results we can set the random seed for Python, Numpy and PyTorch')


args = parser.parse_args()


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

    args.save_path=args.save_path+args.data_name+'_'+args.model_name+'_'+args.TAG
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

    modelpath=os.path.join(args.save_path, 'model_best.pt')
    model_ft.load_state_dict(torch.load(modelpath))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    test_model(model_ft, dataloaders, class_names, criterion, args.batchsize)

    if 'test' in dataloaders.keys():
        test_loader=dataloaders['test']
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images.numpy()
        images = images.to(device)

        # get sample outputs
        outputs = model_ft(images)#torch.Size([32, 10])
        if type(outputs) is tuple: #model may output multiple tensors as tuple
                outputs, _ = outputs
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(outputs, 1) #https://pytorch.org/docs/stable/generated/torch.max.html, dim=1, [32,10]->[32]

        on_gpu = torch.cuda.is_available()
        preds = np.squeeze(preds_tensor.numpy()) if not on_gpu else np.squeeze(preds_tensor.cpu().numpy()) #to numpy array list
        #preds = np.squeeze(preds_tensor.cpu().numpy())

        vistestresult(images, labels, preds, class_names, args.save_path)

        #Start accuracy evaluation
        test_loss, test_acc = evaluate(model_ft, dataloaders['test'], criterion, device)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

        images, labels, probs = get_predictions(model_ft, dataloaders['test'], device)
        pred_labels = torch.argmax(probs, 1)
        plot_confusion_matrix(labels, pred_labels)

        corrects = torch.eq(labels, pred_labels)

        #get all of the incorrect examples and sort them by descending confidence in their prediction
        incorrect_examples = []

        for image, label, prob, correct in zip(images, labels, probs, corrects):
            if not correct:
                incorrect_examples.append((image, label, prob))

        incorrect_examples.sort(reverse = True, key = lambda x: torch.max(x[2], dim = 0).values)
        
        N_IMAGES = 25
        plot_most_incorrect(incorrect_examples, N_IMAGES)


#plot the examples the model got wrong and was most confident about.
def plot_most_incorrect(incorrect, n_images):

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (20, 10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim = 0)
        ax.imshow(image.view(28, 28).cpu().numpy(), cmap = 'bone')
        ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n' \
                     f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    fig.savefig('./outputs/most_incorrect.png')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
def plot_confusion_matrix(labels, pred_labels):
    
    fig = plt.figure(figsize = (10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = range(10));
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.savefig('./outputs/confusion_matrix.png')

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import torch.nn.functional as F
def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


if __name__ == '__main__':
    main()
