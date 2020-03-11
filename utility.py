# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:07:00 2020

@File			:   utility.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Utility functions
@Released		:	
@Updated		:  
    
"""
# Ajouter le temps d'éxecution

import torch
from torchvision import datasets, transforms

# Function to load the datasets 
# Inspired from : https://www.kaggle.com/vincentman0403/pytorch-v0-3-1b-on-mnist-by-lenet (consulted on 07/03/2020)
def load_data(dataset, train_batch_size=64, test_batch_size=1000) :
    """
    \Description : Load the dataset
    \Args : 
        dataset : dataset to load
        train_batch_size : size of the training batch size
        test_batch_size : size og the test batch size
    \Output : 
        train_loader : loader of the train batch
        test_loader : loader of the test batch
    """
    
    if dataset == "MNIST" :
        # Fetch training data
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=True, download=True,
                               transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])),
                                batch_size=train_batch_size, shuffle=True)

        # Fetch test data
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, 
                               transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])),
                                batch_size=test_batch_size, shuffle=True)
    
    elif dataset == "CIFAR10" :
        # Fetch training data
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=True, download=True,
                                 transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])),
                                batch_size=train_batch_size, shuffle=True)

        # Fetch test data
        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=False, 
                                 transform=transforms.Compose([
                                         transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])),
                                batch_size=test_batch_size, shuffle=True)
    
    else :
        raise ValueError("Invalid dataset name. Either choose 'MNIST' or 'CIFAR10'")
            
    return (train_loader, test_loader)



