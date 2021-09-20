import matplotlib.pyplot as plt
import numpy as np
import os

def imshow(inp, title=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(10,10))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    fig.savefig('./outputs/torchimage.png')

# helper function to un-normalize and display an image
def imshowhelper(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image, permute the axes according to the values given

def visbatchimage(images, labels, classes):
    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshowhelper(images[idx])
        ax.set_title(classes[labels[idx]])
    fig.savefig('./outputs/torchonebatchimage.png')

def vistestresult(images, labels, preds, classes, path='./outputs/'):
    #images.numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
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