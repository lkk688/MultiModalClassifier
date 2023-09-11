from __future__ import print_function, division
from scipy.fft import fft, ifft, fftfreq, fftshift
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

from TorchClassifier.Datasetutil.Visutil import visfirstimageinbatch, vistestresult, matplotlib_imshow, plot_most_incorrect, plot_confusion_matrix
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.Datasetutil.Imagenetdata import loadjsontodict, dict2array, preprocess_image, preprocess_imagecv2
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel, createImageNetmodel
from TorchClassifier.TrainValUtils import create_model, test_model, postfilter, \
    model_inference, inference_singleimage, inference_batchimage, collect_incorrect_examples, getclass_newnames

model = None 
device = None
# import logger

os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/' #setting the environment variable

#Tiny Imagenet evaluation
#python myTorchEvaluator.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' 
# --data_path "/data/cmpe249-fa22/ImageClassData" --model_name 'resnet50'
# --checkpoint 'outputs/tiny-imagenet-200_resnet50_0328/checkpoint.pth.tar'
# --classmap 'TorchClassifier/Datasetutil/tinyimagenet_idmap.json'
#image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"

#imagenet_blurred
#python myTorchEvaluator.py --data_name 'imagenet_blurred' --data_type 'trainonly' 
# --data_path "/data/cmpe249-fa22/ImageClassData" --model_name 'resnet50'
# --model_type 'ImageNet'
# --classmap 'TorchClassifier/Datasetutil/imagenet1000id2label.json'
#image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"

#imagenet_blurred
#python myTorchEvaluator.py --data_name 'imagenet_blurred' --data_type 'trainonly' 
# --data_path "/data/cmpe249-fa22/ImageClassData" --model_name 'deit_base_patch16_224'
# --model_type 'ImageNet' --torchhub 'facebookresearch/deit:main'
# --classmap 'TorchClassifier/Datasetutil/imagenet1000id2label.json'
#Test Accuracy (Overall): 80% (40219/49997)
# Test Loss: 0.895 | Test Acc: 80.44%


parser = configargparse.ArgParser(description='myTorchClassify')
parser.add_argument('--data_name', type=str, default='imagenet_blurred',
                    help='data name: imagenet_blurred, tiny-imagenet-200, hymenoptera_data, CIFAR10, MNIST, flower_photos')
parser.add_argument('--data_type', default='valonly', choices=['trainonly', 'trainvalfolder', 'traintestfolder', 'torchvisiondataset'],
                    help='the type of data') 
parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/ImageClassData",
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
parser.add_argument('--model_name', default='deit_base_patch16_224',
                    help='the network') #choices=['resnet50', 'mlpmodel1', 'lenet', 'resnetmodel1', 'vggmodel1', 'cnnmodel1']
parser.add_argument('--model_type', default='ImageNet', choices=['ImageNet', 'custom'],
                    help='the network')
parser.add_argument('--torchhub', default='facebookresearch/deit:main',
                    help='the torch hub link')
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
parser.add_argument('--classmap', default='TorchClassifier/Datasetutil/imagenet1000id2label.json', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
# parser.add_argument('--epochs', type=int, default=15,
#                     help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use TPU')
parser.add_argument('--gpuid', default=0, type=int,
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

    model_ft, model_classnames, numclasses, classmap = create_model(args.model_name, args.model_type, args.classmap, args.checkpoint, args.torchhub, device, img_shape)
    model_ft.eval()

    newname="Sports Cars"#classmap['n04285008']
    image_path="/data/cmpe249-fa23/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"#n04285008_497.JPEG"
    inference_singleimage(image_path, model_ft, device, classnames=model_classnames, truelabel=newname, size=args.img_height, top_k=args.topk)
    

    #Load dataset
    dataloaders, dataset_sizes, dataset_classnames, img_shape = loadTorchdataset(args.data_name,args.data_type, args.data_path, args.img_height, args.img_width, args.batchsize)
    print("dataset_classnames:",dataset_classnames) #'n12768682' as names
    print("model_classnames:",model_classnames) # actual label names
    class_newnames = getclass_newnames(args.model_type, classmap, model_classnames, dataset_classnames)
    #print("class_newnames:",class_newnames)

    criterion = nn.CrossEntropyLoss()
    if 'val' in dataloaders.keys():
        val_loader=dataloaders['val']
        # obtain one batch of validation images
        images, labels = next(iter(val_loader)) #[32, 3, 224, 224]
        # Create a grid from the images and show them
        img_grid = torchvision.utils.make_grid(images)
        matplotlib_imshow(img_grid, one_channel=False)

        #(batchsize, topk)
        np_indices, np_probs, batchresults = inference_batchimage(images, model_ft, device, classnames=class_newnames, truelabel=labels, size=args.img_height, top_k=args.topk)

        vistestresult(images, labels, np_indices[:,0], class_newnames, args.save_path)
        #save 'torchtestresultimage.png' under save_path
        collect_incorrect_examples(images, labels, np_indices, args.topk, classnames=class_newnames)


        #Start complete accuracy evaluation
        test_loss, test_accuracy, labels, probs = test_model(model_ft, dataloaders, class_newnames, criterion, args.batchsize, key = 'val', device=device)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy:.2f}%')
        plot_confusion_matrix(labels, probs)




if __name__ == '__main__':
    main()
