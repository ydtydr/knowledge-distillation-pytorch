"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# class sub_cifar100(torch.utils.data.Dataset):
#     def __init__(self,subclass_indices,transform=None):
#         self.subclass_indices = subclass_indices
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.subclass_indices)
    
#     def __getitem__(self,idx):
#         sample = trainset[self.subclass_indices[idx]]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
    
def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    trainset = torchvision.datasets.CIFAR100(root='./data-cifar100', train=True,
        download=True, transform=train_transformer)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devset = torchvision.datasets.CIFAR100(root='./data-cifar100', train=False,
        download=True, transform=dev_transformer)
    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params, subclasses):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    trainset = torchvision.datasets.CIFAR100(root='./data-cifar100', train=True,
        download=True, transform=train_transformer)

    devset = torchvision.datasets.CIFAR100(root='./data-cifar100', train=False,
        download=True, transform=dev_transformer)
    
    class sub_cifar100(torch.utils.data.Dataset):
        def __init__(self,subclass_indices,base,transform=None):
            self.subclass_indices = subclass_indices
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.subclass_indices)

        def __getitem__(self,idx):
            sample = self.base[self.subclass_indices[idx]]
            if self.transform:
                sample = self.transform(sample)
            return sample
    
    train_subclass_indices = list(map(lambda x:x[0],list(filter(lambda x:x[1][1] in subclasses, enumerate(trainset,0)))))
    subclass_trainset = sub_cifar100(train_subclass_indices,trainset)
    
    dev_subclass_indices = list(map(lambda x:x[0],list(filter(lambda x:x[1][1] in subclasses, enumerate(devset,0)))))
    subclass_devset = sub_cifar100(dev_subclass_indices,devset)
    
#     print(len(dev_subclass_indices))

    trainloader = torch.utils.data.DataLoader(subclass_trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    devloader = torch.utils.data.DataLoader(subclass_devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl