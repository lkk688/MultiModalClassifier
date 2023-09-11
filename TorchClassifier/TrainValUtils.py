from enum import Enum
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from TorchClassifier.Datasetutil.Visutil import imshow
import numpy as np
import time
from tqdm import tqdm

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

            #y_pred, _ = model(x)
            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_model(model, dataloaders, class_names, criterion, batch_size, key='test', device='cuda', topk=5):
    numclasses = len(class_names)
    top_k = min(topk, numclasses)
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(numclasses))
    class_total = list(0. for i in range(numclasses))

    model.eval()

    if key in dataloaders.keys():
        test_loader=dataloaders[key]
    else:
        print(f"{key} dataset not available")
        return

    # iterate over test data
    batchindex = 0
    batch_time = AverageMeter('Batch time')
    end = time.time()
    labels = []
    probs = []
    print("Total len of test loader:", len(test_loader))
    pbar = tqdm(desc='Evaluation:', total = len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            batchindex = batchindex +1
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            labels.append(target.cpu())
            probs.append(pred.cpu())

            # calculate test accuracy for each object class
            for i in range(batch_size):
                if i<len(target.data):#the actual batch size of the last batch is smaller than the batch_size
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
            
            pbar.update(1)
    
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

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
    
    test_accuracy=100. * np.sum(class_correct) / np.sum(class_total)
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        test_accuracy,
        np.sum(class_correct), np.sum(class_total)))
    return test_loss, test_accuracy, labels, probs
    
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk) #5
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) #[85, 5]
        pred = pred.t() #[5, 128]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def visualize_model(model, dataloaders, class_names, num_images=6, device="cuda"):
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

def visualize_result(model, dataloaders, classes, key='val', device='cuda'):
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

from TorchClassifier.Datasetutil.Imagenetdata import loadjsontodict, dict2array, preprocess_image, preprocess_imagecv2
from TorchClassifier.Datasetutil.Visutil import visfirstimageinbatch, plot_most_incorrect
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel, createImageNetmodel
import os
def create_model(model_name, model_type, classmap, checkpoint=None, torchhub=None, device="cuda", img_shape=[2, 224, 224]):
    #Load class map
    classmap=loadjsontodict(classmap)
    #Create model
    if model_type == "ImageNet":
        model_ft, classnames, numclasses, preprocess = createImageNetmodel(model_name, torchhub)
        model_ft = model_ft.to(device)
        if classnames is None:
            classnames=dict2array(classmap)
            numclasses=len(classmap)
    else:
        classnames=dict2array(classmap)
        numclasses=len(classmap)

        model_ft = createTorchCNNmodel(model_name, numclasses, img_shape)
        model_ft = model_ft.to(device)
        if checkpoint and os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=device)
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
            print(f"Loading checkpoint: {checkpoint}")
    return model_ft, classnames, numclasses, classmap

def getclass_newnames(model_type, classmap, model_classnames, dataset_classnames):
    print("Dataset Class names:", len(dataset_classnames))
    if model_type == "ImageNet":
        if model_classnames:
            class_newnames = model_classnames #1000 class
        else:
            class_newnames = dataset_classnames #1000 class
    else:
        class_newnames=[]
        for name in dataset_classnames: #from the dataset
            newname=classmap[name]
            class_newnames.append(newname)
    return class_newnames

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