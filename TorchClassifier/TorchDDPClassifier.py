import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
# setting the environment variable
os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/'


def create_data_loader_cifar10():
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='/data/cmpe249-fa22/torchvisiondata/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=10, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='/data/cmpe249-fa22/torchvisiondata/', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=10)
    return trainloader, testloader


def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 1
    num_of_batches = len(trainloader)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            images, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(
            f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')

    print('Finished Training')


if __name__ == '__main__':
    start = time.time()
    PATH = './cifar_net.pth'
    trainloader, testloader = create_data_loader_cifar10()

    net = torchvision.models.resnet50(True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # Batch size should be divisible by number of GPUs
        # DataParallel is single-process, multi-thread, and only works on a single machine
        net = nn.DataParallel(net)
    
    net = net.cuda()
    start_train = time.time()
    train(net, trainloader)
    end_train = time.time()
    # save
    torch.save(net.state_dict(), PATH)
    # test
    #test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
        Train 1 epoch {seconds_train:.2f} seconds")
