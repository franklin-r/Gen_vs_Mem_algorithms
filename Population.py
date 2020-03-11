# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:13:31 2020

@File			:   Population.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Population for genetic algorithms
@Released		:	
@Updated		: 
    
"""

from random import choice
import Individual as ind



# Define the Individual
class Population() :
    """
    \Description : Population for genetic algorithms
    \Attributes :
        dataset : dataset used
        size : population's size
        NL_set : set of possible values for the number of hidden layers
        NF_set : set of possible values for the number of feature maps
        lr_set : set of possible values for the learning rate
        momentum_set : set of possible values for the momentum
        pop : list of individuals in the population
    """
    def __init__(self, dataset, size, NL_set, NF_set, lr_set, momentum_set) :
        """
        \Description: Build a population
        \Args : 
            dataset : dataset used
            size : population's size
            NL_set : set of possible values for the number of hidden layers
            NF_set : set of possible values for the number of feature maps
            lr_set : set of possible values for the learning rate
            momentum_set : set of possible values for the momentum
        \Outputs : None
        """
        self.dataset = dataset      # Dataset
        self.size = size            # Size of the population
        
        # Those are sets of values to choose from to create an indivudal
        self.NL_set = NL_set
        self.NF_set = NF_set
        self.lr_set = lr_set
        self.momentum_set = momentum_set
        
        self.pop = []           # List of the individuals
        
        
        
        # Create an initial populations
        for i in range(0, self.size) :
            # Append a random individual
            self.pop.append(
                    ind.Individual(dataset=self.dataset,
                                   NL=choice(self.NL_set),
                                   NF=choice(self.NF_set),
                                   lr=choice(self.lr_set),
                                   momentum=choice(self.momentum_set)))
            
            
            
        
        
        
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
        print("NL set : {}".format(self.NL_set))
        print("NF set : {}".format(self.NF_set))
        print("lr set : {}".format(self.lr_set))
        print("momentum set : {}".format(self.momentum_set))
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        