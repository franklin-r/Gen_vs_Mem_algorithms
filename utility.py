# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:56:39 2020

@File			:   utility.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Utility functions to save the data
@Released		:	
@Updated		: 
    
"""

import csv

# TO DO
def save_pareto_front(popul, data) :
    
    # Iterate through the whole population
    for model in popul.pop :
        # Save the data of the current model in the Pareto frontier
        model_results = [len(data)+1, model.inaccuracy, model.time, model.NL, model.NF, model.lr, model.mom]
        
        data.append(model_results)
    # end for model
    
    # Append a separation between each Pareto frontier
    data.append(["XXX", "XXX", "XXX", "XXX", "XXX", "XXX"])
    
# end save_pareto_front()


def write_data_to_csv(filename, header, data) :
    """
    \Description: Write the data corresponding to header in a csv file called filename
    \Args : 
        filename    : the name of the file
        header      : a list containing the name if the columns
        data        : a matrix containinf the data
    \Outputs : None
    """ 
    row_list = [header] + data
    
    with open(filename, 'a', newline='') as file :
        writer = csv.writer(file)
        writer.writerows(row_list)
