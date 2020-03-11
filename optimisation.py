# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:46:12 2020

@File			:   Optimisation.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Optimisation techniques for CNN architecture
@Released		:	
@Updated		: 
    
"""

import Individual as ind
import utility as ut

import CNN

'''
# Grid search optimisation for optimal number of layer, number of feature maps, 
# learning rate and momentum
def grid_search(dataset, NL_set, NF_set, lr_set, momentum_set) :
    """
    \Description: Test all the possible cominations of hyper-paramters
    \Args : 
        dataset : datset used
        NL_set : set of possible values for the number of hidden layers
        NF_set : set of possible values for the number of feature maps
        lr_set : set of possible values for the learning rate
        momentum_set : set of possible values for the momentum
    \Outputs : None
    """
    for NL in NL_set :
        for NF in NF_set :
            for lr in lr_set :
                for momentum in momentum_set :
                    # Create individual
                    indiv = ind.Individual(dataset=dataset,
                                           NL=NL,
                                           NF=NF,
                                           lr=lr,
                                           momentum=momentum)
                    
                    # Test the individual
                    
                    
                    
                    # Check if the individual is optimal
                    
                    
                # end for momentum
            # end for lr
        # end for NF
    # end for NL
'''    
    
    
    
    
    
'''    
def gen_algo(popul, gen_max) :
    
    for gen in range(0, gen_max) :
        # Evaluation of current population
        for model in popul.pop :
            model.evaluate_model()
'''            
        
        # Selection of individuals (best and random not best). Place them in a new population P_next
        
        # Crossover. Randomly select individuals in P_next. Place offspring in P_child
        
        # Mutation of individuals in P_child
        
        # Update of current population. Current population is the union of P_next and P_child
        
    # return pop
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    