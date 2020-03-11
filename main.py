# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:29:58 2020

@File			:   main.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Main program
@Released		:	
@Updated		: 
    
"""

import CNN
import utility as ut

if __name__ == "__main__" :
    
    model = CNN.CNN(dataset="MNIST", NL=3, NF=3, lr=0.01, mom=0.5)
    
    model.printCNN()
    
    train_loader, test_loader = ut.load_data("MNIST", train_batch_size=64, test_batch_size=1000)
    
    model.evaluate_model(train_loader=train_loader, test_loader=test_loader, epochs=1, train_batch_size=64, test_batch_size=1000)
    
    