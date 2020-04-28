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
        
    '''
    # Sets of possible hyper-parameters (sets for grid search)
    NL_set = [i for i in range(8, 12)]
    NF_set = [i for i in range(3, 6)]
    lr_set = [0.1, 0.01, 0.001]
    mom_set = [0.75, 0.8, 0.85, 0.9]
    '''

    # Sets of possible hyper-parameters (sets for gen/mem algo)
    NL_set = [i for i in range(8, 13)]
    NF_set = [i for i in range(3, 6)]
    lr_set = [0.1, 0.01, 0.001]
    mom_set = [0.75, 0.8, 0.85, 0.9]

    ''' 
    # ------------------------ GRID SEARCH SCENARIO ------------------------ #
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=0, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
    # Load the data
    curr_pop.load_data(train_batch_size=64, test_batch_size=512)
    
    # Grid search
    opt.grid_search(curr_pop, epochs=10)
    '''
    
    
    # --------------------- GENETIC ALGORITHM SCENARIO --------------------- #
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=9, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
    
    # Load the data
    curr_pop.load_data(train_batch_size=64, test_batch_size=512)
    
    # Genetic algorithm for no exact objective values
    opt.gen_algo(popul=curr_pop, epochs=10, nb_best=4, pm=0.25, gen_max=10)

    # Genetic algortihm for exact objective values
    #opt.gen_algo(popul=curr_pop, epochs=10, nb_best=4, pm=0.25, gen_max=15, inaccuracy=70, time=1.8)
    '''
    
    # --------------------- MEMETIC ALGORITHM SCENARIO --------------------- #
    # Create a population
    curr_pop = Pop.Population(dataset="CIFAR10", 
                              size=9, 
                              NL_set=NL_set, 
                              NF_set=NF_set, 
                              lr_set=lr_set, 
                              mom_set=mom_set)
 
    # Load the data
    curr_pop.load_data(train_batch_size=64, test_batch_size=512)
    
    # Memetic algorithm for no exact objective values
    #opt.mem_algo(popul=curr_pop, epochs=10, nb_best=4, pm=0.25, radius=0.05, nb_neighb=3, gen_max=10)

    # Memetic algorithm for exact objective values
    opt.mem_algo(popul=curr_pop, epochs=10, nb_best=4, pm=0.25, radius=0.05, nb_neighb=3, gen_max=15, inaccuracy=70, time=1.8)
    ''' 
# end main 
