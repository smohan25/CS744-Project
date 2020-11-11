#!/usr/bin/python
import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import argparse
import model as mdl
import datetime
import time
import math

device = "cpu"
torch.set_num_threads(4)
batch_size = 256 # batch for one node
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


parser = argparse.ArgumentParser(description='Data-parallel training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--master-ip', default='tcp://10.10.1.1:6585', type=str,
                    help='master ip')
parser.add_argument('--num-nodes', type=int, help='total number of nodes')

'''
def average_gradients(model):
    """                                                                                                                                                                    
    Averages the gradients using all reduce                                                                                                                                
    """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
'''

def topHalf(tensor):
    # In dimension 0
    size = tensor.size(0)
    if size % 2 == 0:
        return tensor.narrow(0, 0, size/2)
    else:
        return tensor.narrow(0, 0, size/2 + 1)
    
def botHalf(tensor):
    size = tensor.size(0)
    if size % 2 == 0:
        return tensor.narrow(0, size/2, size/2)
    else:
        return tensor.narrow(0, size/2+1, size/2)
    
def average_gradients2(model, rank, size):
    for param in model.parameters():
        original_val = param.grad.data
        original_val = torch.tensor([1, 2, 3, 4])
        resVec = original_val
        steps = math.log(size, 2)
        d = 2
        for i in range(int(steps)):
            if (rank % d) < d/2:
                dest = rank + d/2
                sendVector = topHalf(resVec)
                recvVector = torch.zeros_like(botHalf(resVec))
                concatVector = torch.zeros_like(sendVector)
                dist.send(sendVector, dest)
                print "Sent"
                dist.recv(recvVector, dest)
                print "Recv"
                res = torch.cat((concatVector, recvVector))
                resVec = resVec.add(res)
            else:
                dest = rank - d/2
                sendVector = botHalf(resVec)
                recvVector = torch.zeros_like(topHalf(resVec))
                concatVector = torch.zeros_like(sendVector)
                dist.recv(recvVector, src=dest)
                print "Recv"
                dist.send(sendVector, dest)
                print "Sent"
                res = torch.cat((recvVector, concatVector))
                resVec = resVec.add(res)
            #print "Dest is", dest, sendVector
            #dist.send(sendVector, dest)
            #dist.recv(recvVector, src=dest)
            if (rank % d) < d/2:
                resVec = botHalf(resVec)
            else:
                resVec = topHalf(resVec)
            d *= 2

        # Perform all-gather...
        d = size
        for i in range(int(steps)):
            recvVector = torch.zeros_like(resVec)
            if (rank % d) < d/2:
                dest = rank + d/2
                dist.send(resVec, dest)
                dist.recv(recvVector, src=dest)
                resVec = torch.cat((recvVector, resVec))
            else:
                dest = rank - d/2
                dist.recv(recvVector, src=dest)
                dist.send(resVec, dest)
                resVec = torch.cat((resVec, recvVector))
            d /= 2
        print "Value is", resVec
        exit()
        

def train_model(model, train_loader, optimizer, criterion, epoch, args):
    """                                                                                                                                                                    
    model (torch.nn.module): The model created to train                                                                                                                    
    train_loader (pytorch data loader): Training data loader                                                                                                               
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD                                                                                             
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network                                                                                              
    epoch (int): Current epoch number                                                                                                                                      
    args: program arguments                                                                                                                                                
    """

    distUrl = "tcp://" + args.master_ip + ":6585"
    torch.distributed.init_process_group('gloo', init_method=distUrl, timeout=datetime.timedelta(0, 1800),
                                         world_size=args.num_nodes, rank=args.rank, store=None, group_name='')

    print("Init process done....")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx == 1:
            t1_start = time.time()
        
        # Initialize gradients to be zero                                                                                                                                  
        optimizer.zero_grad()

        # Get the output vector                                                                                                                                            
        output = model(data)

        # Calculate the loss                                                                                                                                               
        loss = criterion(output, target)

        # Find the gradients
        loss.backward()

        #average_gradients(model)
        average_gradients2(model, args.rank, args.num_nodes)

        # Update the weights                                                                                                                                               
        optimizer.step()

        if batch_idx == 9:
            t1_stop = time.time()
            print("Avg iteration time: {} seconds"
            .format((t1_stop-t1_start)/9))

        if (batch_idx + 1) % 20 == 0:
            print("Loss: {}".format(loss.cpu().data.item()))

    print("Finished training....")
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    sampler = DistributedSampler(training_set, num_replicas=args.num_nodes, rank=args.rank, shuffle=True)


    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size/args.num_nodes,
                                                    sampler=sampler,
                                                    pin_memory=True)

    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch                                                                                                                                       
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, args)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
