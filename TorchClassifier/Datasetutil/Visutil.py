import matplotlib.pyplot as plt
import numpy as np
import os
import torch

#ImageNet means and stds
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    fig = plt.figure(figsize=(10,10))
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    inp = img.numpy().transpose((1, 2, 0))
    mean = np.array(pretrained_means)
    std = np.array(pretrained_stds)
    inp = std * inp + mean #range0-1
    #print(max(inp[:,0,0]))
    inp = np.clip(inp, 0, 1)
    if one_channel:
        plt.imshow(inp, cmap="Greys")
    else:
        plt.imshow(inp)
        
def imshow(inp, title=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(10,10))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(pretrained_means)
    std = np.array(pretrained_stds)
    inp = std * inp + mean #range0-1
    #print(max(inp[:,0,0]))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp) #The first two dimensions (M, N) define the rows and columns of the image
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    fig.savefig('./outputs/torchimage.png')

def visfirstimageinbatch(img_batch, batchresults, classnames=None, truelabel=None):
    idx=0 # first image in the batch
    if truelabel is not None:
        if type(truelabel) == list:
            titles=truelabel[idx]+"\n"
        else:
            titles=truelabel+"\n"
    else:
        titles=""
    for i in range(len(batchresults[idx])): #multiple predicted results
        dict = batchresults[idx][i]
        if dict:
            first_class=dict['class_idx'] #0 means top1 result
            first_confidence=dict['confidence']
            if classnames is not None and first_class<len(classnames):
                title=f"Predicted class: {classnames[first_class]}, confidence: {first_confidence:.4f} \n"
            else:
                title=f"Predicted idx: {first_class}, confidence: {first_confidence:.4f} \n"
            titles += title
    #img=np.transpose(img_batch.cpu()[idx], (1, 2, 0)) #already done in imshow
    imshow(img_batch.cpu()[idx],title=titles)

# helper function to un-normalize and display an image
def imshowhelper(img):
    #img = img / 2 + 0.5  # unnormalize
    inp = np.transpose(img.numpy(), (1, 2, 0))#img.transpose((1, 2, 0))
    mean = np.array(pretrained_means)
    std = np.array(pretrained_stds)
    inp = std * inp + mean #range0-1
    #print(max(inp[:,0,0]))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    #plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image, permute the axes according to the values given



def visbatchimage(images, labels, classes):
    #images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
        imshowhelper(images[idx])
        ax.set_title(classes[labels[idx]])
    fig.savefig('./outputs/torchonebatchimage.png')

def vistestresult(images, labels, preds, classes, path='./outputs/'):
    #images.numpy()
    print("labels:", labels)
    print("preds:", preds)
    print("preds len:", len(preds))
    print("classes:", classes)
    

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    numfigs = min(20, len(preds))
    for idx in np.arange(numfigs): #2*10 Image array
        ax = fig.add_subplot(2, int(numfigs/2), idx+1, xticks=[], yticks=[])
        imshowhelper(images.cpu()[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]), color=("green" if preds[idx]==labels[idx].item() else "red"))
    figsavepath=os.path.join(path, 'torchtestresultimage.png')
    fig.savefig(figsavepath)#'./outputs/torchtestresultimage.png')


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def visimagelistingrid(images,path='./outputs/', normalize = False): #images is a list
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (10, 10)) #plt.figure()
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image = images[i]
        if normalize:
            image = normalize_image(image)
        if(len(image.shape) == 3):
            #imgdata=np.squeeze(image) # works for grey image
            imgdata=image.permute(1, 2, 0) #put the channels as the last dimension
        elif(len(image.shape) == 2):
            imgdata=image
        else:
            print("Higher dimensional data")
        ax.imshow(imgdata.cpu().numpy())#, cmap = 'bone')
        #ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap = 'bone')
        #images[i] shape: [1,28,28]
        #imshow expects images to be structured as (rows, columns) for grayscale data and (rows, columns, channels) and possibly (rows, columns, channels, alpha) values for RGB(A) data
        ax.axis('off')
    figsavepath=os.path.join(path, 'visimagelistingrid.png')
    fig.savefig(figsavepath)

#plot the examples the model got wrong and was most confident about.
def plot_most_incorrect(incorrect, n_images, classnames):

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (20, 10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image, true_idx, pred_idx = incorrect[i]
        # true_prob = probs[true_label]
        # incorrect_prob, incorrect_label = torch.max(probs, dim = 0)
        #ax.imshow(image.view(imageshape[1],imageshape[2]).cpu().numpy(), cmap = 'bone')
        img=image.permute(1,2,0).cpu() #convert from (3,32,32)
        ax.imshow(img, cmap = 'bone')
        if classnames is not None:
            true_class=classnames[true_idx.item()]#get value from tensor
            pred_class=classnames[pred_idx.item()]
            ax.set_title(f'true label: {true_idx}, class: {true_class}\n' \
                        f'pred label: {pred_idx}, class: {pred_class}')
        else:
            ax.set_title(f'true label: {true_idx}\n' \
                        f'pred label: {pred_idx}')
        # ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n' \
        #              f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    fig.savefig('./outputs/most_incorrect.png')

#pip install -U scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
def plot_confusion_matrix(labels, pred_labels, maxlen=200):
    if len(labels)>maxlen:
        labels=labels[:maxlen]
        pred_labels=pred_labels[:maxlen]
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    #cm = ConfusionMatrixDisplay(cm, display_labels = range(10))
    cm = ConfusionMatrixDisplay(cm)
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.savefig('./outputs/confusion_matrix.png')
