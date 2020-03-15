# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:13:31 2020

@File			:   Population.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Population for genetic algorithms
@Released		:	
@Updated		: 
    
"""

import CNN
import torch
from torchvision import datasets, transforms
from random import choice



# Define the Individual
class Population() :
    """
    \Description : Population for genetic algorithms
    \Attributes :
        dataset             : dataset used
        size                : population's size (must be an even number)
        NL_set              : set of possible values for the number of hidden layers
        NF_set              : set of possible values for the number of feature maps
        lr_set              : set of possible values for the learning rate
        mom_set             : set of possible values for the momentum
        pop                 : list of individuals in the population
        train_loader        : train loader
        test_loader         : test_loader
        train_batch_size    : size of the training batch
        test_batch_size     : size of the testing batch
    """
    def __init__(self, dataset, size, NL_set, NF_set, lr_set, mom_set, indiv_list=[]) :
        """
        \Description: Build a population of random individual
        \Args : 
            dataset     : dataset used
            size        : population's size
            NL_set      : set of possible values for the number of hidden layers
            NF_set      : set of possible values for the number of feature maps
            lr_set      : set of possible values for the learning rate
            mom_set     : set of possible values for the momentum
            indiv_list  : list of individuals
        \Outputs : None
        """
        
        self.dataset = dataset      # Dataset
        self.size = size            # Size of the population
        
        # Those are sets of values to choose from to create an indivudal
        self.NL_set = NL_set
        self.NF_set = NF_set
        self.lr_set = lr_set
        self.mom_set = mom_set
        
        self.pop = indiv_list           # List of the individuals in the population
        self.train_loader = None        # Train loader
        self.test_loader = None         # Test loader
        self.train_batch_size = 0       # Size of the training batch
        self.test_batch_size = 0        # Size of the testing batch
        
        
        if indiv_list == [] :
            # Create a random initial populations
            for i in range(0, self.size) :
                # Append a random individual
                '''
                self.pop.append(
                        CNN.CNN(dataset=self.dataset,
                                NL=choice(self.NL_set),
                                NF=choice(self.NF_set),
                                lr=choice(self.lr_set),
                                mom=choice(self.mom_set)).cuda())
                '''
                self.pop.append(
                        CNN.CNN(dataset=self.dataset,
                                NL=choice(self.NL_set),
                                NF=choice(self.NF_set),
                                lr=choice(self.lr_set),
                                mom=choice(self.mom_set)))
            # end for i
        # end if indiv_list == []
    # end __init__()    
            
        
    # Print the caracteristics fo the population
    def printPopulation(self) :
        """
        \Description: Print the population's info
        \Args : None
        \Outputs : None
        """
        print("POPULATION")
        print("Dataset : {}".format(self.dataset))
        print("Population size : {}".format(self.size))
        print("Set of number of layers : {}".format(self.NL_set))
        print("Set of number of feature maps : {}".format(self.NF_set))
        print("Set of learning rate : {}".format(self.lr_set))
        print("Set of momentum : {}".format(self.mom_set))
    # end printPopulation()
    
    
    # Function to load the datasets 
    # Inspired from : https://www.kaggle.com/vincentman0403/pytorch-v0-3-1b-on-mnist-by-lenet (consulted on 07/03/2020)
    def load_data(self, train_batch_size=64, test_batch_size=1000) :
        """
        \Description : Load the dataset
        \Args : 
            train_batch_size : size of the training batch size
            test_batch_size : size og the test batch size
        \Output : 
            train_loader : loader of the train batch
            test_loader : loader of the test batch
        """
        
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        if self.dataset == "MNIST" :
            # Fetch training data
            self.train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('data', train=True, download=True,
                                   transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                           ])),
                                    batch_size=self.train_batch_size, shuffle=True)
    
            # Fetch test data
            self.test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('data', train=False, 
                                   transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                           ])),
                                    batch_size=self.test_batch_size, shuffle=True)
        
        elif self.dataset == "CIFAR10" :
            # Fetch training data
            self.train_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('data', train=True, download=True,
                                     transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])),
                                    batch_size=self.train_batch_size, shuffle=True)
    
            # Fetch test data
            self.test_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('data', train=False, 
                                     transform=transforms.Compose([
                                             transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])),
                                    batch_size=self.test_batch_size, shuffle=True)
        
        else :
            raise ValueError("Invalid dataset name. Either choose 'MNIST' or 'CIFAR10'")

# end class Population        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        