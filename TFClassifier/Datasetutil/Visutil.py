import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np


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
    fig.savefig('25images.png')

def plot9imagesfromtfdataset(image_ds, class_names):
    fig = plt.figure(figsize=(10,10))
    for images, labels in image_ds.take(1):
        for i in range(9):
            print(np.min(images[i]), np.max(images[i])) #0~1
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())#plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    fig.savefig('9images.png')