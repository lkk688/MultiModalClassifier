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

# # PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)

from TorchClassifier.Datasetutil.Visutil import visfirstimageinbatch, vistestresult, matplotlib_imshow, plot_most_incorrect
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.Datasetutil.Imagenetdata import loadjsontodict, dict2array, preprocess_image, preprocess_imagecv2
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel
from TorchClassifier.TrainValUtils import test_model
# from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset #loadtfds, loadkerasdataset, loadimagefolderdataset
# from TFClassifier.myTFmodels.CNNsimplemodels import createCNNsimplemodel
# from TFClassifier.Datasetutil.Visutil import plot25images, plot9imagesfromtfdataset, plot_history
# from TFClassifier.myTFmodels.optimizer_factory import build_learning_rate, setupTensorboardWriterforLR

model = None 
device = None
# import logger

os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/' #setting the environment variable

#Tiny Imagenet evaluation
#python myTOrchEvaluator.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' 
# --data_path "/data/cmpe249-fa22/ImageClassData" --model_name 'resnet50'
# --checkpoint 'outputs/tiny-imagenet-200_resnet50_0328/checkpoint.pth.tar'
# --classmap 'TorchClassifier/Datasetutil/tinyimagenet_idmap.json'
#image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"


parser = configargparse.ArgParser(description='myTorchClassify')
parser.add_argument('--data_name', type=str, default='imagenet_blurred',
                    help='data name: imagenet_blurred, tiny-imagenet-200, hymenoptera_data, CIFAR10, MNIST, flower_photos')
parser.add_argument('--data_type', default='valonly', choices=['trainonly', 'trainvalfolder', 'traintestfolder', 'torchvisiondataset'],
                    help='the type of data') 
parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa22/ImageClassData",
                    help='path to get data') #/Developer/MyRepo/ImageClassificationData
parser.add_argument('--img_height', type=int, default=224,
                    help='resize to img height, 224')
parser.add_argument('--img_width', type=int, default=224,
                    help='resize to img width, 224')
parser.add_argument('--topk', type=int, default=5,
                    help='show top k results')
parser.add_argument('--save_path', type=str, default='./outputs/',
                    help='path to save the model')
# network
parser.add_argument('--model_name', default='resnet50', choices=['mlpmodel1', 'lenet', 'resnetmodel1', 'vggmodel1', 'cnnmodel1'],
                    help='the network')
parser.add_argument('--model_type', default='ImageNet', choices=['ImageNet', 'custom'],
                    help='the network')
parser.add_argument('--checkpoint', default='outputs/tiny-imagenet-200_resnet50_0328/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--arch', default='Pytorch', choices=['Tensorflow', 'Pytorch'],
#                     help='Model Name, default: Pytorch.')
# parser.add_argument('--learningratename', default='warmupexpdecay', choices=['fixedstep', 'fixed', 'warmupexpdecay'],
#                     help='learning rate name')
# parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adam'],
#                     help='select the optimizer')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batchsize', type=int, default=32,
                    help='batch size')
parser.add_argument('--classmap', default='TorchClassifier/Datasetutil/tinyimagenet_idmap.json', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
# parser.add_argument('--epochs', type=int, default=15,
#                     help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use TPU')
parser.add_argument('--gpuid', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--ddp', default=False, type=bool,
                    help='Use multi-processing distributed training.')
parser.add_argument('--MIXED_PRECISION', type=bool, default=False,
                    help='use MIXED_PRECISION')
parser.add_argument('--TAG', default='0328',
                    help='setup the experimental TAG to differentiate different running results')
parser.add_argument('--reproducible', type=bool, default=False,
                    help='get reproducible results we can set the random seed for Python, Numpy and PyTorch')


args = parser.parse_args()


def postfilter(indices, probs, classnames=None, min_threshold=0.1):
    batchsize=indices.shape[0]
    resultlen=indices.shape[1]
    batchresults=[]
    for batch in range(batchsize):
        topkresult=[] #for single image
        for i in range(resultlen):
            oneresult={}
            if probs[batch][i] > min_threshold:
                idx=indices[batch][i]
                oneresult['class_idx']= idx
                oneresult['confidence']= probs[batch][i]
                if classnames is not None and len(classnames)>idx:
                    oneresult['classname']=classnames[idx]
            topkresult.append(oneresult)
        batchresults.append(topkresult)
    return batchresults

def model_inference(model, img_batch, top_k):
    output = model(img_batch) #torch.Size([batchsize, classlen])
    if type(output) is tuple: #model may output multiple tensors as tuple
        output, _ = output
    output_prob = output.softmax(-1) #convert logits to probability for dim = -1
    output_prob, indices = output_prob.topk(top_k) #[256,batchsize]
    np_indices = indices.cpu().numpy() #(batchsize, 5)
    np_probs = output_prob.cpu().numpy()
    return np_indices, np_probs

def inference_singleimage(image_path, model, device, classnames=None, truelabel=None, size=224, top_k=5, min_threshold=0.1):
    img_batch = preprocess_imagecv2(image_path, imagesize=size)
    img_batch = img_batch.to(device)
    
    with torch.no_grad():
        np_indices, np_probs = model_inference(model, img_batch, top_k)
        batchresults = postfilter(np_indices, np_probs, classnames=classnames, min_threshold=min_threshold)
    visfirstimageinbatch(img_batch, batchresults, classnames, truelabel)

def inference_batchimage(img_batch, model, device, classnames=None, truelabel=None, size=224, top_k=5, min_threshold=0.1):
    img_batch = img_batch.to(device)
    with torch.no_grad():
        np_indices, np_probs = model_inference(model, img_batch, top_k)
        batchresults = postfilter(np_indices, np_probs, classnames=classnames, min_threshold=min_threshold)
    #visfirstimageinbatch(img_batch, batchresults, classnames, truelabel)
    return np_indices, np_probs, batchresults

from torchvision.models import get_model, get_model_weights, get_weight, list_models
def createImageNetmodel(model_name):
    # Step 1: Initialize model with the best available weights
    weights_enum = get_model_weights(model_name)
    weights = weights_enum.IMAGENET1K_V1
    #print([weight for weight in weights_enum])
    #weights = get_weight("ResNet50_Weights.IMAGENET1K_V2")#ResNet50_Weights.DEFAULT
    currentmodel=get_model(model_name, weights=weights)#weights="DEFAULT"
    #currentmodel.eval()
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()#preprocess.crop_size
    classes = weights.meta["categories"]
    # Step 3: Apply inference preprocessing transforms
    #batch = preprocess(img).unsqueeze(0)
    return currentmodel, classes, preprocess

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
        loc = 'cuda:{}'.format(args.gpuid)
        device = torch.device(loc if torch.cuda.is_available() else "cpu")

    else:
        print("No GPU and TPU enabled")

    img_shape=[3, args.img_height, args.img_width] #[channels, height, width] in pytorch

    #Load class map
    classmap=loadjsontodict(args.classmap)
    #Create model
    if args.model_type == "ImageNet":
        model_ft, classnames, preprocess = createImageNetmodel(args.model_name)
        model_ft = model_ft.to(device)
        numclasses=len(classnames)
    else:
        classnames=dict2array(classmap)
        numclasses=len(classmap)

        model_ft = createTorchCNNmodel(args.model_name, numclasses, img_shape)
        model_ft = model_ft.to(device)
        if args.checkpoint and os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            state_dict_key = ''
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict_key = 'state_dict'
                elif 'model' in checkpoint:
                    state_dict_key = 'model'
            model_state=checkpoint[state_dict_key]
            size=model_state['fc.bias'].shape
            print(f"Output size in model: {size[0]}, numclasses: {numclasses}")
            model_ft.load_state_dict(model_state)
            print(f"Loading checkpoint: {args.checkpoint}")
    model_ft.eval()

    newname=classmap['n04285008']
    image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"#n04285008_497.JPEG"
    inference_singleimage(image_path, model_ft, device, classnames=classnames, truelabel=newname, size=args.img_height, top_k=args.topk)
    

    #Load dataset
    dataloaders, dataset_sizes, class_names, img_shape = loadTorchdataset(args.data_name,args.data_type, args.data_path, args.img_height, args.img_width, args.batchsize)
    print("Class names:", len(class_names))
    if args.model_type == "ImageNet":
        class_newnames = classnames #1000 class
    else:
        class_newnames=[]
        for name in class_names: #from the dataset
            newname=classmap[name]
            class_newnames.append(newname)

    newname=classmap['n04285008']
    image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"#n04285008_497.JPEG"
    inference_singleimage(image_path, model_ft, device, classnames=class_newnames, truelabel=newname, size=args.img_height, top_k=args.topk)
    criterion = nn.CrossEntropyLoss()
    

    if 'val' in dataloaders.keys():
        val_loader=dataloaders['val']
        # obtain one batch of validation images
        images, labels = next(iter(val_loader)) #[32, 3, 224, 224]
        # Create a grid from the images and show them
        img_grid = torchvision.utils.make_grid(images)
        matplotlib_imshow(img_grid, one_channel=False)

        # Default log_dir argument is "runs" - but it's good to be specific
        # torch.utils.tensorboard.SummaryWriter is imported above
        # writer = SummaryWriter('outputs/experiment_1')
        # # Write image data to TensorBoard log dir
        # writer.add_image('ExperimentImages', img_grid)
        # writer.flush()
        # To view, start TensorBoard on the command line with:
        #   tensorboard --logdir=runs
        # ...and open a browser tab to http://localhost:6006/

        #(batchsize, topk)
        np_indices, np_probs, batchresults = inference_batchimage(images, model_ft, device, classnames=class_newnames, truelabel=labels, size=args.img_height, top_k=args.topk)

        vistestresult(images, labels, np_indices[:,0], class_newnames, args.save_path)
        collect_incorrect_examples(images, labels, np_indices, args.topk, classnames=class_newnames)


        #Start complete accuracy evaluation
        test_loss, test_accuracy, labels, probs = test_model(model_ft, dataloaders, class_newnames, criterion, args.batchsize, key = 'val', device=device)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy:.2f}%')
        plot_confusion_matrix(labels, probs)
        

        

def collect_incorrect_examples(images, labels, np_indices, topk, classnames=None):
    if topk>1:
        top1=np_indices[:,0] #[batchsize, topk]
    top1=torch.from_numpy(top1)
    corrects = torch.eq(labels, top1)#compare tensor
    #get all of the incorrect examples and sort them by descending confidence in their prediction
    incorrect_examples = []

    for image, label, prob, correct in zip(images, labels, top1, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse = True, key = lambda x: torch.max(x[2], dim = 0).values)
    
    N_IMAGES = min(len(incorrect_examples),25)
    plot_most_incorrect(incorrect_examples, N_IMAGES, classnames)



#pip install -U scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
def plot_confusion_matrix(labels, pred_labels):
    
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    #cm = ConfusionMatrixDisplay(cm, display_labels = range(10))
    cm = ConfusionMatrixDisplay(cm)
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.savefig('./outputs/confusion_matrix.png')


import torch.nn.functional as F
def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            #y_pred, _ = model(x)
            y_pred = model(x)

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
