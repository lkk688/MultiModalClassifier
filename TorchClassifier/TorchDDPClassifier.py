import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os


os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

MACHINENAME='HPC'#'AlienwareX51'#'HPC'

UseTorchDataParallel=False #True
UseTorchDistributedDataParallel=True
MIXEDPRECISION=False

# setting the environment variable
if MACHINENAME=='HPC':
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/'
    DATAPATH='/data/cmpe249-fa22/torchvisiondata/'
elif MACHINENAME=='AlienwareX51':
    DATAPATH='/DATA8T/Datasets/torchvisiondata'



def create_data_loader_cifar10():
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root=DATAPATH, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATAPATH, train=False,
                                           download=True, transform=transform)

    if UseTorchDistributedDataParallel:
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)                                                  
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=0, pin_memory=True)
        
        test_sampler =DistributedSampler(dataset=testset, shuffle=True)                                         
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, sampler=test_sampler, num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=10, pin_memory=True)

    
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=10)
    return trainloader, testloader


def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 1
    num_of_batches = len(trainloader)

    if MIXEDPRECISION:
        fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):  # loop over the dataset multiple times

        if UseTorchDistributedDataParallel:
            #In distributed mode, calling the data_loader.sampler.set_epoch() method at the beginning of each epoch before creating the DataLoader iterator to make shuffling work properly across multiple epochs
            trainloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            images, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            if MIXEDPRECISION:
                # forward
                with torch.cuda.amp.autocast():
                    outputs = net(images)
                    loss = criterion(outputs, labels)

                # mixed precision training 
                # backward + optimizer step
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
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

#for Torch DPP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl", #nccl,gloo (works in HPC)
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

if __name__ == '__main__':
    start = time.time()
    PATH = './outputs/cifar_net.pth'

    if UseTorchDistributedDataParallel:
        init_distributed()

    trainloader, testloader = create_data_loader_cifar10()

    net = torchvision.models.resnet50(True)#download to /home/lkk/.cache/torch/hub/checkpoints

    if UseTorchDataParallel:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # Batch size should be divisible by number of GPUs
            # DataParallel is single-process, multi-thread, and only works on a single machine
            net = nn.DataParallel(net)
    
    net = net.cuda()
    if UseTorchDistributedDataParallel:
        # Convert BatchNorm to SyncBatchNorm. 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

        local_rank = int(os.environ['LOCAL_RANK'])
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

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
    
    if is_main_process():
        # do that ….
        # save, load models, download data etc….
        print("In the main process")

#Launch script using torch.distributed.launch or torch.run
#python -m torch.distributed.launch --nproc_per_node=4 ./TorchClassifier/TorchDDPClassifier.py

#ref: https://theaisummer.com/distributed-training-pytorch/
