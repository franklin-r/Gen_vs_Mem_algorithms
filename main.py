# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:29:58 2020

@File			:   main.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Main program
@Released		:	
@Updated		: 
    
"""

import torch
import Population as Pop

if __name__ == "__main__" :
    
     # Provide seed for the pseudorandom number generator
    torch.manual_seed(123)
    
    # Sets of possible hyper-parameters
    NL_set = range(3, 5)
    NF_set = range(3, 6)
    lr_set = [0.1, 0.01, 0.001, 0.0001]
    mom_set = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    # Create a population
    curr_pop = Pop.Population(dataset="MNIST", size=1, NL_set=NL_set, NF_set=NF_set, lr_set=lr_set, mom_set=mom_set)
    
    # Print population's info
    curr_pop.printPopulation()
    
    print("==================================")
    
    # Print individuals' info
    for indiv in curr_pop.pop :
        indiv.printCNN()
        
    print("==================================")
    
    # Load the dataset
    curr_pop.load_data(train_batch_size=64, test_batch_size=1000)
    
    # Evaluate the models
    for indiv in curr_pop.pop :
        indiv.evaluate_model(train_loader=curr_pop.train_loader,
                             test_loader=curr_pop.test_loader,
                             epochs=10,
                             train_batch_size=curr_pop.train_batch_size,
                             test_batch_size=curr_pop.test_batch_size)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        