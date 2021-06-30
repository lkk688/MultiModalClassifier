import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import os


def plot25images(images, labels, class_names):
    fig = plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        # The labels happen to be arrays, which is why you need the extra index
        #plt.xlabel(class_names[labels[i][0]])
        #print(labels[i].shape)
        #print(labels[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()
    fig.savefig('./outputs/25images.png')

def plot9imagesfromtfdataset(image_ds, class_names):
    fig = plt.figure(figsize=(10,10))
    for images, labels in image_ds.take(1):
        for i in range(9):
            print(np.min(images[i]), np.max(images[i])) #0~1
            ax = plt.subplot(3, 3, i + 1)
            #plt.imshow(images[i].numpy())#
            if np.max(images[i])<=1:
                imgnp=images[i].numpy()#normalized data 0-1
            else:
                imgnp=images[i].numpy().astype("uint8") #0-255
            plt.imshow(imgnp)
            plt.title(class_names[labels[i]])
            plt.axis("off")
    fig.savefig('./outputs/9images.png')


def plot_history(history, metric, val_metric, path='./outputs/traininghistory.pdf'):
    acc = history.history[metric]
    val_acc = history.history[val_metric]

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([min(plt.ylim()), 1])
    plt.grid(True)
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title('Training and Validation Loss')
    plt.show()
    print(path)
    fig.savefig(path+'/traininghistory.pdf')#'./outputs/traininghistory.pdf')