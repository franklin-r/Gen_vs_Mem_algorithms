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

def shape_pareto_front(pareto_frontiers) :
    """
    \Description: Shape the pareto front to save it in a .csv file
    \Args : 
        pareto_frontiers    : matrix containing the Pareto-optimal individuals of each iteration to shape
    \Outputs : 
        shaped_data : the data shaped for .csv file
    """ 
    # Tha data shaped to be written in .csv file afterward
    shaped_data = []
    
    # Iterate through the whole Pareto frontier
    for i in range(0, len(pareto_frontiers)) :
        for model in pareto_frontiers[i] :
            # Save the data of the current model in the Pareto frontier
            if type(model.inaccuracy).__name__ == "float" :
                shaped_data.append(
                        [i + 1,
                        model.inaccuracy,
                        model.time,
                        model.chromosome["NL"],
                        model.feat_maps_seq,
                        model.chromosome["lr"],
                        model.chromosome["mom"]])

            # If model.inaccuracy is a tensor (case of grid search)
            else :
                shaped_data.append(
                        [i + 1, 
                        model.inaccuracy.item(), 
                        model.time, 
                        model.chromosome["NL"],
                        model.feat_maps_seq, 
                        model.chromosome["lr"], 
                        model.chromosome["mom"]])
            # end if
        # end for model
    # end for it
    
    return shaped_data
# end shape_pareto_front()


def write_data_to_csv(filename, header, data) :
    """
    \Description: Write the data corresponding to header in a .csv file called filename
    \Args : 
        filename    : the name of the file
        header      : a list containing the name of the columns
        data        : a matrix containing the data
    \Outputs : None
    """ 
    row_list = [header] + data
    
    with open(filename, 'a', newline='') as file :
        writer = csv.writer(file)
        writer.writerows(row_list)
# end write_data_to_csv()
