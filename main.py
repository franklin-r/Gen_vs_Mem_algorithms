# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:29:58 2020

@File			:   main.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Main program
@Released		:	
@Updated		: 
    
"""

import Population as Pop
import optimisation as opt

if __name__ == "__main__" :
    
    '''
    # Sets of possible hyper-parameters
    NL_set = [i for i in range(8, 16)]
    NF_set = [i for i in range(3, 6)]
    lr_set = [0.1, 0.01, 0.001, 0.0001]
    mom_set = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    '''
    
    # Sets of possible hyper-parameters
    NL_set = [i for i in range(4, 6)]
    NF_set = [i for i in range(3, 5)]
    lr_set = [0.1, 0.01]
    mom_set = [0.7, 0.75]
    
    
    # ------------------------ GRID SEARCH SCENARIO ------------------------ #
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=0, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
    # Load the data
    curr_pop.load_data(train_batch_size=64, test_batch_size=1000)
    
    # Grid search
    opt.grid_search(curr_pop)
    
    
    '''
    # --------------------- GENETIC ALGORITHM SCENARIO --------------------- #
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=12, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
    # Load the data
    curr_pop.load_data(train_batch_size=64, test_batch_size=1000)
    
    # Genetic algorithm
    opt.gen_algo(popul=curr_pop, gen_max=10, nb_best=2, pm=0.25)
    
    
    # --------------------- MEMETIC ALGORITHM SCENARIO --------------------- #
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=12, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
    # Load the data
    curr_pop.load_data(train_batch_size=64, test_batch_size=1000)
    
    # Genetic algorithm
    opt.mem_algo(popul=curr_pop, gen_max=10, nb_best=2, pm=0.25)
    
    
    '''
    '''
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=0, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
    # Print population's info
    curr_pop.printPopulation()
    
    print("==================================")
    
    # Print individuals' info
    for indiv in curr_pop.pop :
        indiv.printCNN()
        
    print("==================================")
    
    
    
    
    
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=1, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
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
    '''   
    
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
