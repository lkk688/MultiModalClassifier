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
import onnx

# # PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)

from TorchClassifier.Datasetutil.Visutil import visfirstimageinbatch, vistestresult, matplotlib_imshow, plot_most_incorrect, plot_confusion_matrix
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.Datasetutil.Imagenetdata import loadjsontodict, dict2array, preprocess_image, preprocess_imagecv2
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel, createImageNetmodel
from TorchClassifier.TrainValUtils import test_model
# from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset #loadtfds, loadkerasdataset, loadimagefolderdataset
# from TFClassifier.myTFmodels.CNNsimplemodels import createCNNsimplemodel
# from TFClassifier.Datasetutil.Visutil import plot25images, plot9imagesfromtfdataset, plot_history
# from TFClassifier.myTFmodels.optimizer_factory import build_learning_rate, setupTensorboardWriterforLR

model = None 
device = None
# import logger

os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/' #setting the environment variable


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
parser.add_argument('--model_name', default='deit_base_patch16_224', choices=['resnet50', 'mlpmodel1', 'lenet', 'resnetmodel1', 'vggmodel1', 'cnnmodel1'],
                    help='the network')
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
    #img_batch = preprocess_imagecv2(image_path, imagesize=size)
    img_batch = preprocess_image(image_path, imagesize=size)
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

def prepare_inputs(dataloader, device, numbatch=10):
    """load sample inputs to device"""
    inputs = []
    for batch in dataloader:
        if type(batch) is torch.Tensor:
            batch_d = batch.to(device)
            batch_d = (batch_d, )
            inputs.append(batch_d)
        else:
            batch_d = []
            for x in batch:
                assert type(x) is torch.Tensor, "input is not a tensor"
                batch_d.append(x.to(device))
            batch_d = tuple(batch_d)
            inputs.append(batch_d)
        if len(inputs)>numbatch:
            return inputs
    return inputs

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
        model_ft, classnames, numclasses, preprocess = createImageNetmodel(args.model_name, args.torchhub)
        model_ft = model_ft.to(device)
        if classnames is None:
            classnames=dict2array(classmap)
            numclasses=len(classmap)
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

    newname="Sports Cars"#classmap['n04285008']
    image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"#n04285008_497.JPEG"
    #inference_singleimage(image_path, model_ft, device, classnames=classnames, truelabel=newname, size=args.img_height, top_k=args.topk)
    
    #inputs = prepare_inputs(dataloaders['val'], device)
    inputs = preprocess_image(image_path, imagesize=args.img_height)
    inputs = inputs.to(device)
    ONNX_FILE_PATH = os.path.join(args.save_path, args.model_name+'.onnx')
    with torch.no_grad():
        # torch.onnx.export(model_ft, inputs, ONNX_FILE_PATH, input_names=['input'],
        #           output_names=['output'], export_params=True)
        torch.onnx.export(model_ft,
                          inputs,
                          ONNX_FILE_PATH,
                          verbose=True,
                          opset_version=13,
                          do_constant_folding=True)
    
    #check the model
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    main()
