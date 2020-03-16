# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:46:12 2020

@File			:   Optimisation.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Optimisation techniques for CNN architecture
@Released		:	
@Updated		: 
    
"""

import Population as Pop
import CNN
from random import sample, shuffle, randint, random, choice


def pop_next_iter(model, popul) :
    """
    \Description : Determine the population of best individuals for the next iteration
    \Args : 
        model   : the model to test  
        popul   : the population to test
    \Outputs : None
    """
    new_pop = []            # List containing the individual of the next iteration
    add = False
    
    
    for i in range(0, len(popul.pop)) :
        # doit parcourir toute la population
        
        #If model dominates another one
        if model.inaccuracy <= popul.pop[i].inaccuracy and model.time <= popul.pop[i].time :
            add = True              # This model can be added to the nex iteration
                                    # and we do not add popul.pop[i]
            
        # If model is dominated by another one
        elif model.inaccuracy > popul.pop[i].inaccuracy and model.time > popul.pop[i].time :
            add = False                         # We do not add model  
            new_pop += popul.pop[i:]            # We add the rest of popul.pop
            break                               # We stop for this model
        # end if
    # end for i
    
    # Once the population has been looped over, we check if we can add model to the next iteration
    if add == True :
        new_pop.append(model)
    # end if
    
    # Update the population for the next iteration
    popul.pop = new_pop
    
    # Udate the size of the population
    popul.size = len(new_pop)
# end pop_next_iter()



# Grid search optimisation for optimal number of layer, number of feature maps, 
# learning rate and momentum
def grid_search(popul) :
    """
    \Description: Test all the possible combinations of hyper-parameters and find the best
    \Args : 
        popul : an empty population
    \Outputs : None
    """    

    for NL in popul.NL_set :
        for NF in popul.NF_set :
            for lr in popul.lr_set :
                for mom in popul.mom_set :
                    # Create the individual
                    model = CNN.CNN(dataset=popul.dataset,
                                    NL=NL,
                                    NF=NF,
                                    lr=lr,
                                    mom=mom)
                    
                    # Evaluate the individual
                    model.evaluate_model()
                    
                    # Check if the individual is optimal
                    pop_next_iter(model,popul)                    
                    
                # end for momentum
            # end for lr
        # end for NF
    # end for NL
# end grid_search
    
    
    
def eval_pop(popul) :
    """
    \Description : Evaluate the current population
    \Args : 
        popul   : the population of individual to evaluate
        nb_best : number of best individuals to select for crossover 
    \Outputs : None
    """
    inaccuracies = []                       #List containing the inaccuracy attributes of each model
    times = []                              #List containing the inaccuracy attributes of each model

    # Evaluation of the current population
    for model in popul.pop :
        model.evaluate_model()
        inaccuracies.append(model.inaccuracy)       # Add the inaccuracy of each model to the list
        times.append(model.time)                    # Add the time of each model to the list
    # end for model
    
    max_inaccuracy = max(inaccuracies)
    max_time = max(times)   
        
    # Compute the fitness of each individual    
    for model in popul.pop :
        # Normalize the values by the max and compute the fitness which is the product of both normalized scores
        model.fitness = (model.inaccuracy / max_inaccuracy) * (model.time / max_time)
    # end for model 
    
    # Sort the model according to their fitness
    popul.pop.sort(key=lambda x : x.fitness)
# end eval_pop()
    
    
def selection(popul, nb_best) :
    """
    \Description : Select the best individuals of a population as well as randopm individuals from the rest of the population
    \Args : 
        popul   : the population of individual to evolve
        nb_best : number of best individuals to select (ensure that ((popul.size // 2) - nb_best) is a multiple of 4)
    \Outputs : 
        pop_next : The population of next generation
    """
    # Select the nb_best best individuals
    selected_indiv = popul.pop[0:nb_best]
    
    # Select random individuals from the rest of the population
    # The population must be half full in order to have room for the children
    selected_indiv.extend(sample(popul.pop[nb_best:], (popul.size // 2) - len(selected_indiv)))
    
    # Create the corresponding population
    pop_next = Pop.Population(dataset=popul.dataset, 
                              size=len(selected_indiv), 
                              NL_set=popul.NL_set, 
                              NF_set=popul.NF_set,
                              lr_set=popul.lr_set, 
                              mom_set=popul.mom_set, 
                              indiv_list=selected_indiv)
    
    return pop_next
# end selection()
    

def crossover(pop_next) : 
    """
    \Description :  Randomly select pairs of individuals and crossover them to produce a child.
                    Crossover is a 1-point crossover.
    \Args : 
        pop_next    : the population to crossover
    \Outputs : 
        pop_child : The population containing the children
    """
    # Shuffle the individuals to insert randomness
    shuffle(pop_next.pop)            
    
    # Create the couples
    couples = [pop_next.pop[i*2 : (i+1)*2] for i in (len(pop_next.pop) // 2)]   

    children = []          # Contains the children     
    
    for couple in couples :
        
        # Select the genes that the child heritates
    
        # Make a list from the keys in chromosome dictionary
        genes = list(couple[0].chromosome.keys())   
        
        # Randomly select a cut gene 
        # Before this gene, the child heritates from parent 1's genes, after from parent 2's
        cut_gene = choice(genes)
        
        child_gene = [couple[0].chromosome.get(val) for val in genes[:genes.index(cut_gene)]] + \
                         [couple[1].chromosome.get(val) for val in genes[genes.index(cut_gene):]]
        
        # Create the child from the new genes
        children.append(CNN.CNN(dataset=pop_next.dataset, 
                                NL=child_gene[0],            
                                NF=child_gene[1], 
                                lr=child_gene[2], 
                                mom=child_gene[3]))
    # end for couple
    
    # Create the corresponding population
    pop_child = Pop.Population(dataset=pop_next.dataset, 
                               size=pop_next.size, 
                               NL_set=pop_next.NL_set, 
                               NF_set=pop_next.NF_set,
                               lr_set=pop_next.lr_set, 
                               mom_set=pop_next.mom_set,
                               indiv_list=children)
     
    return pop_child
# end crossover()
    

def mutation(pop_child, pm) :
    """
    \Description :  Randomly apply a mutation to one of the genes of the individuals in pop_child
    \Args : 
        pop_child   : the population to mutate
        pm          : probability for an individual to mutate
    \Outputs : None
    """
    for indiv in pop_child.pop :
        if random() <= pm :
            # Select a random gene to mutate
            gene = choice(list(indiv.chromosome))
            
            # Modify the value of the gene with a random value from the set of possibility
            if gene == "NL" :
                indiv.chromosome["NL"] = choice(pop_child.NL_set)
            
            elif gene == "NF" :
                indiv.chromosome["NF"] = choice(pop_child.NF_set)
                
            elif gene == "lr" :
                indiv.chromosome["lr"] = choice(pop_child.lr_set)
                
            elif gene == "mom" :
                indiv.chromosome["mom"] = choice(pop_child.mom_set)
            # end if gene
        # end random()
    # end for
# end mutation() 
    
  
def gen_algo(popul, gen_max, nb_best, pm) :
    """
    \Description : Apply a genetic algorithm to a population
    \Args : 
        popul   : the population of individual to evolve
        gen_max : max number of generations
        nb_best : number of best individuals to select for crossover 
        pm      : probability for an individual to mutate
    \Outputs : None
    """
    
    for gen in range(0, gen_max) :
        
        # --- EVALUATION ---
        eval_pop(popul)
        
        # --- SELECTION --- 
        pop_next = selection(popul, nb_best)   
        
        # --- CROSSOVER ---
        pop_child = crossover(pop_next)
        
        # --- MUTATION ---
        mutation(pop_child, pm)
        
        # --- UPDATE OF THE CURRENT POPULATION ---  
        # The current population is the union of pop_next and pop_child's population
        popul.pop = pop_next.pop + pop_child.pop
        
        # Delete intermediate populations to free space
        del pop_next
        del pop_child
    # end for gen
# end gen_algo()
    
 
# --- TO DO ---
def mem_algo(popul, gen_max, nb_best, pm) :
    """
    \Description : Apply a memetic algorithm to a population
    \Args : 
        popul   : the population of individual to evolve
        gen_max : max number of generations
        nb_best : number of best individuals to select for crossover 
        pm      : probability for an individual to mutate
    \Outputs : None
    """
    
    for gen in range(0, gen_max) :
        
        # --- EVALUATION ---
        eval_pop(popul)
        
        # --- SELECTION --- 
        pop_next = selection(popul, nb_best)   
        
        # --- CROSSOVER ---
        pop_child = crossover(pop_next)
        
        # --- MUTATION ---
        mutation(pop_child, pm)
        
        # Update of current population. 
        # --- UPDATE OF THE CURRENT POPULATION ---  
        # The current population is the union of pop_next and pop_child's population
        popul.pop = pop_next.pop + pop_child.pop
        
        # Delete intermediate populations to free space
        del pop_next
        del pop_child
    # end for gen
# end gen_algo()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    