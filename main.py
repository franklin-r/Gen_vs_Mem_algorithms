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
#import CNN

if __name__ == "__main__" :
    
    #model = CNN.CNN(dataset="MNIST", NL=3, NF=3, lr=0.01, mom=0.5)
    
    #model.printCNN()
    
    #train_loader, test_loader = ut.load_data("MNIST", train_batch_size=64, test_batch_size=1000)
    
    #model.evaluate_model(train_loader=train_loader, test_loader=test_loader, epochs=1, train_batch_size=64, test_batch_size=1000)
    
    NL_set = range(3, 5)
    NF_set = range(3, 6)
    lr_set = [0.1, 0.01, 0.001, 0.0001]
    mom_set = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    
    curr_pop = Pop.Population(dataset="MNIST", size=1, NL_set=NL_set, NF_set=NF_set, lr_set=lr_set, mom_set=mom_set)
    
    
    print("debug")
    
    curr_pop.printPopulation()
    
    print("==================================")
    
    for indiv in curr_pop.pop :
        indiv.printCNN()
        
    print("==================================")
    
    curr_pop.load_data(train_batch_size=64, test_batch_size=1000)
    
    for indiv in curr_pop.pop :
        indiv.evaluate_model(train_loader=curr_pop.train_loader,
                             test_loader=curr_pop.test_loader,
                             epochs=1,
                             train_batch_size=curr_pop.train_batch_size,
                             test_batch_size=curr_pop.test_batch_size)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        