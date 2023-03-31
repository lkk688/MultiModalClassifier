from torchvision import datasets, models, transforms
import os
import torch
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from TorchClassifier.Datasetutil.Visutil import imshow, visbatchimage, visimagelistingrid
BATCH_SIZE = 32
IMG_height = 180
IMG_width = 180
#In PyTorch, images are represented as [channels, height, width]

# percentage of training set to use as validation
valid_size = 0.2
# number of subprocesses to use for data loading
num_workers = 0 #4  # 0


# Select a random subset of data and corresponding labels
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
# Extract a random subset of data
#images, labels = select_n_random(training_set.data, training_set.targets)

def datanormalization():
    # convert data to a normalized torch.FloatTensor
    # ref: https://pytorch.org/docs/stable/torchvision/transforms.html
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#Normalize a tensor image with mean and standard deviation.
    #     ])

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        #transforms.RandomRotation(10),
        transforms.ToTensor(), #converts the image from a PIL image into a PyTorch tensor.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def datatransforms(mean, std, imagesize=28, training=True):
    if training==True:
        datatransform = transforms.Compose([
                    #transforms.RandomRotation(5, fill=(0,)),
                    transforms.RandomCrop(imagesize, padding = 2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [mean], std = [std])
                                ])
    else:
        datatransform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [mean], std = [std])
                                     ])
    return datatransform

def imagenetdatatransforms(training=True, imagesize=224):
    #All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    if training==True:
        datatransform = transforms.Compose([
                    transforms.Resize(imagesize),
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomCrop(imagesize, padding = 10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
                                ])
    else:
        datatransform = transforms.Compose([
                    transforms.Resize(imagesize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
                                     ])
    return datatransform


# def dataargumation():
#     # Data augmentation and normalization for training
#     # Just normalization for validation
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#     return data_transforms

def dataargumation():
    # Data augmentation and normalization for training
    # Just normalization for validation
    imagesize=IMG_height
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(imagesize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #Imagenet mean and std
        ]),
        'val': transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #Imagenet mean and std
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

    # mydata_transforms = dataargumation()

    # if type == 'trainvalfolder':
    #     # data_dir = 'data/hymenoptera_data' ='/DataDisk1/ImageClassificationData/hymenoptera_data'
    #     datapath = os.path.join(path, name) #data path name is constructed by the input data path and the dataset name
    #     image_datasets = {x: datasets.ImageFolder(os.path.join(datapath, x),
    #                                               mydata_transforms[x])
    #                       for x in ['train', 'val']}
    #     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
    #                                                   shuffle=True, num_workers=num_workers)
    #                    for x in ['train', 'val']}
    #     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #     class_names = image_datasets['train'].classes

    #     # Get a batch of training data
    #     # torch.Size([32, 3, 224, 224])
    #     inputs, classes = next(iter(dataloaders['train']))
    #     imagetensorshape = list(inputs.shape)  # torch.Size to python list
    #     imageshape = imagetensorshape[1:]

    #     # Make a grid from batch
    #     out = torchvision.utils.make_grid(inputs)
    #     imshow(out, title=[class_names[x] for x in classes])

    #     return dataloaders, dataset_sizes, class_names, imageshape
    if type == 'trainvalfolder':
        return loadimagefolderdataset(name, path, split=['train', 'val'])
    elif type == 'traintestfolder':
        return loadimagefoldertraintestdataset(name, path, split=['train', 'test'])
    elif type =='trainonly':
        return loadimagefoldertrainonlydataset(name, path, split=['train'])
    elif type == 'torchvisiondataset':
        return loadtorchvisiondataset(name, path)

def loadimagefoldertrainonlydataset(name, path, split=['train']):
    data_transform = datapreprocess()

    datapath = os.path.join(path, name)
    train_dir = os.path.join(datapath, split[0])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    # print out some data stats
    num_train = len(train_data)
    print('Num training images: ', num_train)
    
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

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)#dataiter.next()
    imagetensorshape = list(images.shape)  # torch.Size to python list
    imageshape = imagetensorshape[1:]

    class_names = train_data.classes
    print('Number of classes: ', len(class_names))
    print('Classes: ', class_names)


    visbatchimage(images, labels, class_names)

    dataloaders = {'train': train_loader,
                    'val': valid_loader}
    dataset_sizes = {'train': len(train_idx), 'val': len(
        valid_idx)}

    return dataloaders, dataset_sizes, class_names, imageshape


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
    images, labels = next(dataiter)#dataiter.next()
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


def loadtorchvisiondataset(name, path):
    datapath = os.path.join(path, name) #data path name is constructed by the input data path and the dataset name
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    # choose the training and test datasets
    if name == 'CIFAR10':
        train_data = datasets.CIFAR10(datapath, train=True,
                                      download=True, transform=datanormalization())  # Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz
        # Extracting data/cifar-10-python.tar.gz to data
        test_data = datasets.CIFAR10(datapath, train=False,
                                     download=True, transform=datanormalization())
    elif name == 'MNIST':
        #
        #automatically download the training set for the MNIST dataset and save it in a folder called datapath. It will create the folder if it does not exist.
        train_data = datasets.MNIST(root = datapath, 
                            train = True, 
                            download = True)
        #To calculate the means and standard deviations we get the actual data (the images) using the .data. attribute of our training data, convert them into floating point numbers and then use the built in mean and std functions to calculate the mean and standard deviation
        mean = train_data.data.float().mean() / 255
        std = train_data.data.float().std() / 255
        print(f'Calculated mean: {mean}') #0.13066048920154572
        print(f'Calculated std: {std}')#0.30810779333114624

        imagesize = IMG_height
        train_data = datasets.MNIST(root = datapath, 
                            train = True, 
                            download = True, 
                            transform = datatransforms(mean, std, imagesize, True))

        test_data = datasets.MNIST(root = datapath, 
                                train = False, 
                                download = True, 
                                transform = datatransforms(mean, std, imagesize, False))
        
        N_IMAGES = 25
        imageslist = [image for image, label in [train_data[i] for i in range(N_IMAGES)]] 
        visimagelistingrid(imageslist)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    #print(len(test_data))  # 10000
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

    #The second option to do split for the validation data
    # train_data, valid_data = torch.utils.data.random_split(train_data, 
    #                                        [int(num_train*0.9), int(num_train*0.1)])
    # train_iterator = torch.utils.data.DataLoader(train_data, 
    #                              shuffle = True, 
    #                              batch_size = BATCH_SIZE)
    # valid_iterator = torch.utils.data.DataLoader(valid_data, 
    #                                 batch_size = BATCH_SIZE)

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)#dataiter.next()
    imagetensorshape = list(images.shape)  # torch.Size to python list
    imageshape = imagetensorshape[1:]

    dataiter = iter(test_loader)
    images, labels =next(dataiter)#dataiter.next()
    imagetensorshape = list(images.shape)  # torch.Size to python list

    visbatchimage(images, labels, class_names)

    dataloaders = {'train': train_loader,
                    'val': valid_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_idx), 'val': len(
        valid_idx), 'test': len(test_data)}

    return dataloaders, dataset_sizes, class_names, imageshape
