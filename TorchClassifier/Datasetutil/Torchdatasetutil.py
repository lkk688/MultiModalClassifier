from torchvision import datasets, models, transforms
import os
import torch
import torchvision

from TorchClassifier.Datasetutil.Visutil import imshow
BATCH_SIZE = 32
IMG_height = 180
IMG_width = 180



def dataargumation():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def loadTorchdataset(name, type, path='/home/lkk/.keras/datasets/flower_photos', img_height=180, img_width=180, batch_size=32):
    global BATCH_SIZE
    BATCH_SIZE=batch_size
    global IMG_height, IMG_width
    IMG_height = img_height
    IMG_width = img_width

    mydata_transforms = dataargumation()

    if type=='trainvalfolder':
        #data_dir = 'data/hymenoptera_data'
        image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                                mydata_transforms[x])
                        for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])

        return dataloaders, dataset_sizes, class_names


