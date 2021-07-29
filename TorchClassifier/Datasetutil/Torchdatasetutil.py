from torchvision import datasets, models, transforms
import os
import torch
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from TorchClassifier.Datasetutil.Visutil import imshow, visbatchimage
BATCH_SIZE = 32
IMG_height = 180
IMG_width = 180

# percentage of training set to use as validation
valid_size = 0.2
# number of subprocesses to use for data loading
num_workers = 4  # 0


def datanormalization():
    # convert data to a normalized torch.FloatTensor
    # ref: https://pytorch.org/docs/stable/torchvision/transforms.html
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#Normalize a tensor image with mean and standard deviation.
    #     ])

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


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

def datapreprocess():
    transform = transforms.Compose([transforms.RandomResizedCrop(IMG_height), 
                                      transforms.ToTensor()])
    return transform

def loadTorchdataset(name, type, path, img_height=180, img_width=180, batch_size=32):
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    global IMG_height, IMG_width
    IMG_height = img_height
    IMG_width = img_width

    mydata_transforms = dataargumation()

    if type == 'trainvalfolder':
        # data_dir = 'data/hymenoptera_data' ='/DataDisk1/ImageClassificationData/hymenoptera_data'
        datapath = os.path.join(path, name)
        image_datasets = {x: datasets.ImageFolder(os.path.join(datapath, x),
                                                  mydata_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                      shuffle=True, num_workers=num_workers)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        # Get a batch of training data
        # torch.Size([32, 3, 224, 224])
        inputs, classes = next(iter(dataloaders['train']))
        imagetensorshape = list(inputs.shape)  # torch.Size to python list
        imageshape = imagetensorshape[1:]

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])

        return dataloaders, dataset_sizes, class_names, imageshape
    elif type == 'trainvalfolder':
        return loadimagefolderdataset(name, path, split=['train', 'val'])
    elif type == 'traintestfolder':
        return loadimagefoldertraintestdataset(name, path, split=['train', 'test'])
    elif type == 'torchvisiondataset':
        return loadtorchvisiondataset(name)

def loadimagefoldertraintestdataset(name, path, split=['train', 'test']):
    data_transform = datapreprocess()

    datapath = os.path.join(path, name)
    train_dir = os.path.join(datapath, split[0])
    test_dir = os.path.join(datapath, split[1])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)
    # print out some data stats
    num_train = len(train_data)
    num_test = len(test_data)
    print('Num training images: ', num_train)
    print('Num test images: ', num_test)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=num_workers)#sampler and shuffle cannot be used at the same time
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True)

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imagetensorshape = list(images.shape)  # torch.Size to python list
    imageshape = imagetensorshape[1:]

    class_names = train_data.classes

    visbatchimage(images, labels, class_names)

    dataloaders = {'train': train_loader,
                    'val': valid_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_idx), 'val': len(
        valid_idx), 'test': len(test_data)}

    return dataloaders, dataset_sizes, class_names, imageshape

def loadimagefolderdataset(name, path, split=['train', 'val']):
    mydata_transforms = dataargumation()

    datapath = os.path.join(path, name)
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        datapath, x), mydata_transforms[x]) for x in split}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x],        batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers) for x in split}
    dataset_sizes = {x: len(image_datasets[x]) for x in split}
    class_names = image_datasets[split[0]].classes  # 'train'

    # Get a batch of training data
    # torch.Size([32, 3, 224, 224])
    inputs, classes = next(iter(dataloaders[split[0]]))
    imagetensorshape = list(inputs.shape)  # torch.Size to python list
    imageshape = imagetensorshape[1:]

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    return dataloaders, dataset_sizes, class_names, imageshape


def loadtorchvisiondataset(name):
    # choose the training and test datasets
    if name == 'CIFAR10':
        train_data = datasets.CIFAR10('data', train=True,
                                      download=True, transform=datanormalization())  # Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz
        # Extracting data/cifar-10-python.tar.gz to data
        test_data = datasets.CIFAR10('data', train=False,
                                     download=True, transform=datanormalization())
        print(len(test_data))  # 10000
        class_names = train_data.classes
        # specify the image classes
        # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        #         'dog', 'frog', 'horse', 'ship', 'truck']

        # obtain training indices that will be used for validation
        num_train = len(train_data)  # 50000
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                                   sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                                   sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                                  num_workers=num_workers)

        # obtain one batch of training images
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        imagetensorshape = list(images.shape)  # torch.Size to python list
        imageshape = imagetensorshape[1:]

        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        imagetensorshape = list(images.shape)  # torch.Size to python list

        visbatchimage(images, labels, class_names)

        dataloaders = {'train': train_loader,
                       'val': valid_loader, 'test': test_loader}
        dataset_sizes = {'train': len(train_idx), 'val': len(
            valid_idx), 'test': len(test_data)}

        return dataloaders, dataset_sizes, class_names, imageshape
