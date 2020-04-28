# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:46:12 2020

@File			:   optimisation.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	Optimisation techniques for CNN architecture
@Released		:	
@Updated		: 
    
"""

import torch
import Population as Pop
import CNN
from random import seed, sample, shuffle, random, choice
from time import perf_counter
from datetime import datetime
import utility as ut
from math import pow, sqrt

def pop_next_iter(popul, model) :
    """
    \Description : Determine the population of best individuals for the next iteration
    \Args : 
        popul   : the population to test
        model   : the model to test
    \Outputs : None
    """
    
    seed(321)               # Set the random seed

    new_pop = []            # List containing the models of the next iteration  
    add = True              # Boolean indicating whether or not to add model to the next iteration
    
    # Iterate through the whole population
    for i in range(0, len(popul.pop)) :
        # If model is dominated by another one
        if ((model.inaccuracy >= popul.pop[i].inaccuracy) and (model.time > popul.pop[i].time)) \
        or ((model.inaccuracy > popul.pop[i].inaccuracy) and (model.time >= popul.pop[i].time)) :
            # We add the rest of the population but not model
            add = False
            new_pop.extend(popul.pop[i:])
            break                               # We stop for this model
        
        # If model dominates another one
        elif ((model.inaccuracy <= popul.pop[i].inaccuracy) and (model.time < popul.pop[i].time)) \
        or   ((model.inaccuracy < popul.pop[i].inaccuracy) and (model.time <= popul.pop[i].time)) :
            # We add model but popul.pop[i]
            add = True
            
        # Any other case, i.e. :
        # model.inaccuracy < popul.pop[i].inaccuracy and model.time > popul.pop[i].time
        # model.inaccuracy > popul.pop[i].inaccuracy and model.time < popul.pop[i].time
        # model.inaccuracy == popul.pop[i].inaccuracy and model.time == popul.pop[i].time
        # falls in the case in which we add both model and popul.pop[i] to the next iteration
        else :
            add = True
            new_pop.append(popul.pop[i])
         # end if   
    # end for i
    
    if add == True :
        new_pop.append(model)
        
    # Update the population for the next iteration
    popul.pop = new_pop[:]
    
    # Udate the size of the population
    popul.size = len(popul.pop)
# end pop_next_iter()



# Grid search optimisation for optimal number of layer, number of feature maps, 
# learning rate and momentum
def grid_search(popul, epochs) :
    """
    \Description: Test all the possible combinations of hyper-parameters and find the best
    \Args : 
        popul   : an empty population
        epochs  : number of epochs
    \Outputs : None
    """    
    # Pareto frontiers to be shaped
    pareto_frontiers = []
    
    # Measure starting time
    start = perf_counter()
    
    for NL in popul.NL_set :
        for NF in popul.NF_set :
            for lr in popul.lr_set :
                for mom in popul.mom_set :
                    # Create the individual
                    if torch.cuda.is_available() :
                        model = CNN.CNN(dataset=popul.dataset,
                                        NL=NL,
                                        NF=NF,
                                        lr=lr,
                                        mom=mom).cuda()
                        
                    else :
                        model = CNN.CNN(dataset=popul.dataset,
                                        NL=NL,
                                        NF=NF,
                                        lr=lr,
                                        mom=mom)
                    # end if

                    # Evaluate the individual
                    model.evaluate_model(train_loader=popul.train_loader,
                                         test_loader=popul.test_loader, 
                                         epochs=epochs, 
                                         train_batch_size=64, 
                                         test_batch_size=512)
                    
                    # Check if the individual is optimal
                    pop_next_iter(popul, model)  
                    
                    # Update the Pareto frontiers
                    pareto_frontiers.append(popul.pop)
                # end for momentum
            # end for lr
        # end for NF
    # end for NL
    
    # Measure starting time
    end = perf_counter()
	
    # Column names
    header = ["Pareto Frontier", "Inaccuracy (%)", "Time (s)", "NL", "NF", "lr", "mom", "Duration (s)"]  

    # Shaped data
    shaped_data = ut.shape_pareto_front(pareto_frontiers)
    
    # Add the duration at the end of the first line
    shaped_data[0].insert(len(shaped_data[1]), end-start)
    
    # Name of the file is the date and time
    filename = datetime.now().strftime("./results/grid_search/%d-%m-%Y_%Hh%M.csv")
    
    # Write the data to the csv file
    ut.write_data_to_csv(filename, header, shaped_data)
    
    # Name of the file is the date and time
    filename = datetime.now().strftime("./results/grid_search/%d-%m-%Y_%Hh%M.txt")

    # Save the best models' caracteristics
    for model in pareto_frontiers[len(pareto_frontiers) - 1] :
        model.printCNN(standard_out="file", filename=filename)
    
# end grid_search
    
    
    
def eval_pop(popul, epochs) :
    """
    \Description : Evaluate the current population
    \Args : 
        popul   : the population of individual to evaluate
        epochs  : number of epochs
    \Outputs : None
    """
    inaccuracies = []                       # List containing the inaccuracy attributes of each model
    times = []                              # List containing the inaccuracy attributes of each model

    # Evaluation of the current population
    for model in popul.pop :
        if model.time == 0.0 :              # If the model has never been trained
            model.evaluate_model(train_loader=popul.train_loader,
                                test_loader=popul.test_loader, 
                                epochs=epochs, 
                                train_batch_size=64, 
                                test_batch_size=512)
        # end if

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
    \Description : Select the best individuals of a population as well as random individuals from the rest of the population
    \Args : 
        popul   : the population of individual to evolve
        nb_best : number of best individuals to select
    \Outputs : 
        pop_next : The population of next generation
    """
    # Select the nb_best best individuals
    selected_indiv = popul.pop[0:nb_best]
   
    # Select 2/3 of the population so that the the last 1/3 is from the crossover
    nb_rand_indiv = (popul.size - len(selected_indiv)) - (popul.size // 3)

    # Select random individuals from the rest of the population
    selected_indiv.extend(sample(popul.pop[nb_best:], nb_rand_indiv))
    
    # Create the corresponding population
    pop_next = Pop.Population(dataset=popul.dataset, 
                              size=len(selected_indiv), 
                              NL_set=popul.NL_set, 
                              NF_set=popul.NF_set,
                              lr_set=popul.lr_set, 
                              mom_set=popul.mom_set, 
                              indiv_list=selected_indiv,
                              train_loader=popul.train_loader,
                              test_loader=popul.test_loader,
                              train_batch_size=popul.train_batch_size,
                              test_batch_size=popul.test_batch_size)
    
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
    couples = [pop_next.pop[i*2 : (i+1)*2] for i in range(0, len(pop_next.pop) // 2)]   

    children = []          # Contains the children     
    
    # Make a list from the keys in chromosome dictionary
    genes = list(pop_next.pop[0].chromosome.keys())   
    
    for couple in couples :
        
        # Select the genes that the child heritates
    
        # Randomly select a cut gene 
        # Before this gene, the child heritates from parent 1's genes, after from parent 2's
        cut_gene = choice(genes)
        
        child1_gene = [couple[0].chromosome.get(val) for val in genes[:genes.index(cut_gene)]] + \
                         [couple[1].chromosome.get(val) for val in genes[genes.index(cut_gene):]]
        
        # Create the child from the new genes
        if torch.cuda.is_available() :
            children.append(CNN.CNN(dataset=pop_next.dataset, 
                                    NL=child1_gene[0],            
                                    NF=child1_gene[1], 
                                    lr=child1_gene[2], 
                                    mom=child1_gene[3]).cuda())
        else :
            children.append(CNN.CNN(dataset=pop_next.dataset, 
                                    NL=child1_gene[0],            
                                    NF=child1_gene[1], 
                                    lr=child1_gene[2], 
                                    mom=child1_gene[3]))
        # end if
    # end for couple
    
    # Create the corresponding population
    pop_child = Pop.Population(dataset=pop_next.dataset, 
                               size=len(children), 
                               NL_set=pop_next.NL_set, 
                               NF_set=pop_next.NF_set,
                               lr_set=pop_next.lr_set, 
                               mom_set=pop_next.mom_set,
                               indiv_list=children,
                               train_loader=pop_next.train_loader,
                               test_loader=pop_next.test_loader,
                               train_batch_size=pop_next.train_batch_size,
                               test_batch_size=pop_next.test_batch_size)
     
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
    
    
def pareto_front(popul) :
    """
    \Description : Return the Pareto frontier of a population
    \Args : 
        popul   : the population 
    \Outputs : 
        pareto_front : the list of individuals of the Pareto frontier
    """
    # Initialize the Pareto frontier with the first individual
    pareto_front = [popul.pop[0]]     
        
    # Iterate through the whole population
    for i in range(1, len(popul.pop)) :    # Start at index 1 because indiv at index 0 is already in the Pareto frontier
        # Boolean indicating whether or not to add an individual in the Pareto frontier
        add = True
       
        # Iterate through the whole Pareto frontier
        j = 0
        while j < len(pareto_front) :
            # If popul.pop[i] is dominated by a Pareto optimal individual
            if ((popul.pop[i].inaccuracy >= pareto_front[j].inaccuracy) and (popul.pop[i].time > pareto_front[j].time)) \
            or ((popul.pop[i].inaccuracy > pareto_front[j].inaccuracy) and (popul.pop[i].time >=  pareto_front[j].time)) :
                # We do not add popul.pop[i]
                add = False
                break           # We stop for this model
            
            # If popul.pop[i] dominates a Pareto optimal individual
            elif ((popul.pop[i].inaccuracy <= pareto_front[j].inaccuracy) and (popul.pop[i].time < pareto_front[j].time)) \
            or   ((popul.pop[i].inaccuracy < pareto_front[j].inaccuracy) and (popul.pop[i].time <= pareto_front[j].time)) :
                # We add popul.pop[i] but remove pareto_front[j]
                add = True
                pareto_front.pop(j)
                j -= 1          # If we do not decrement j, it skips the individual that were after the one removed
            
            # Any other case, i.e. :
            # popul.pop[i].inaccuracy == pareto_front[j].inaccuracy and popul.pop[i].time == pareto_front[j].time
            # popul.pop[i].inaccuracy >= pareto_front[j].inaccuracy and popul.pop[i].time <= pareto_front[j].time
            # popul.pop[i].inaccuracy <= pareto_front[j].inaccuracy and popul.pop[i].time >= pareto_front[j].time
            # falls in the case in which we add both popul.pop[i] and pareto_front[j] to the Pareto front
            else :
                add = True
                # end if

            j += 1
        # end while
        
        if add == True :
            pareto_front.append(popul.pop[i])
        # end if
        
    # end for i
    
    return pareto_front
# end pareto_front()


def check_objectives(pareto_front, inaccuracy, time) : 
    """
    \Description : Check if the objectives are met
    \Args :
        pareto_front    : Pareto frontier
        inaccuracy      : objective inaccuracy
        time            : objective inference time
    \Output : a boolean indicating whether or not the objectives are met
    Note : None
    """

    for model in pareto_front :
        if((model.inaccuracy <= inaccuracy) and (model.time <= time)) :
            return True
    # end for model

    return False

# end check_objectives()

def gen_algo(popul, epochs, nb_best, pm, gen_max, inaccuracy=-1, time=-1) :
    """
    \Description : Apply a genetic algorithm to a population
    \Args : 
        popul       : the population of individual to evolve
        epochs      : number of epochs
        nb_best     : number of best individuals to select for crossover 
        pm          : probability for an individual to mutate
        gen_max     : maximum number of generations
        inaccuracy  : desired inaccuracy. A value of -1 indicates that we do not want an exact value as objective
        time        : desired inference time. A value of -1 indicates that we do not want an exact value as objective
    \Outputs : None
    Note : None
    """
    # Pareto front of each generation
    pareto_frontiers = []
    
    # Measure starting time
    start = perf_counter()
     
    for gen in range(0, gen_max) :

        print("Generation {}".format(gen + 1))
        
        # --- EVALUATION ---
        eval_pop(popul, epochs)
        
        # Extract the current Pareto frontier and update the Pareto frontiers
        pareto_frontiers.append(pareto_front(popul))
        
        # Check whether or not the objectives are met
        if(check_objectives(pareto_frontiers[gen], inaccuracy, time)) :
            break           # Exit the loop if the objectives are met

        # --- SELECTION --- 
        pop_next = selection(popul, nb_best)   
        
        # --- CROSSOVER ---
        pop_child = crossover(pop_next)
        
        # --- MUTATION ---
        mutation(pop_child, pm)
        
        # --- UPDATE OF THE CURRENT POPULATION ---  
        # The current population is the union of pop_next and pop_child's population
        popul.pop = pop_next.pop + pop_child.pop
           
    # end for gen
    
    # Measure ending time
    end = perf_counter()
    
    # Column names
    header = ["Pareto Frontier", "Inaccuracy (%)", "Time (s)", "NL", "NF", "lr", "mom", "Duration (s)"]  

    # Shaped data
    shaped_data = ut.shape_pareto_front(pareto_frontiers)
    
    # Add the duration at the end of the first line
    shaped_data[0].insert(len(shaped_data[1]), end-start)
    
    # Name of the file is the date and time
    filename = datetime.now().strftime("./results/gen_algo/%d-%m-%Y_%Hh%M.csv")
    
    # Write the data to the csv file
    ut.write_data_to_csv(filename, header, shaped_data)
   
    # Name of the file is the date and time
    filename = datetime.now().strftime("./results/gen_algo/%d-%m-%Y_%Hh%M.txt")
    
    # Save the best models' caracteristics
    for model in pareto_frontiers[len(pareto_frontiers) - 1] :
        model.printCNN(standard_out="file", filename=filename)
    
# end gen_algo()
    

def local_search(popul, radius, nb_neighb) :    
    """
    Description : Apply a local search algorithm to an individual of population
    Args : 
        popul       : the population of individual to evolve
        model       : the starting model
        radius      : the radius in which to search for the solution
        nb_neighb   : number of neighbours to test in the radius
    Outputs : None
    Notes : 
        In the case we do not sort the individuals before the local search, we might start to perform local search on 
        "average" individual and replace them with fitter individuals. The probleme lays in the case where the former were in 
        the radius of an optimizable individual.
        After the local search on the "average" individual, it disappears from the radius 
        of an optimizable individual and therefore the latter cannot be optimized anymore.
        The current implementation does not take into account this and a future feature could be added to sort the individuals
        into successive Pareto frontier beforehand if the density of the individuals is too high and such a feature could 
        improve the results. Then we could perform the local search starting from the least fit Pareto frontier            
    """
    # New population after local search
    new_pop = []
    
    # Iterate through the whole population
    for i in range(0, len(popul.pop)) :
        
        # Counter of visited neighbours
        visited_neighb = 0
        
        # Initialize the fittest model of the neighbourhood to the origin model
        fittest_model = popul.pop[i]
        
        # Iterate through the whole population but the fixed individual
        for candidate in (popul.pop[0:i] + popul.pop[i+1:len(popul.pop)]) :
            
            # Stop the local search when enough neighbours has been visited
            if visited_neighb == nb_neighb :
                break
            
            # If candidate is in the quarter circle that defines a better model than popul.pop[i]    
            if ((radius >= sqrt(pow(popul.pop[i].inaccuracy - candidate.inaccuracy, 2) + pow(popul.pop[i].time - candidate.time, 2))) \
            and (((candidate.inaccuracy <= popul.pop[i].inaccuracy) and (candidate.time < popul.pop[i].time)) \
            or  ((candidate.inaccuracy < popul.pop[i].inaccuracy) and (candidate.time <= popul.pop[i].time)))) :
            
                # If candidate is fitter than the fittest of the neighbours
                if (((candidate.inaccuracy <= fittest_model.inaccuracy) and (candidate.time < fittest_model.time)) \
                or ((candidate.inaccuracy < fittest_model.inaccuracy) and (candidate.time <= fittest_model.time))) :
 
                    # Replace the fittest model
                    fittest_model = candidate
                # end if
            # end if
            
            visited_neighb += 1
        # end for candidate
        
        # Add the fittest model to the future population
        new_pop.append(fittest_model)
    # end for i
    
    # Update the population
    popul.pop = new_pop[:]

    # Update the size of popul.pop
    popul.size = len(popul.pop)

# end local_search()
 

def mem_algo(popul, epochs, nb_best, pm, radius, nb_neighb, gen_max, inaccuracy=-1, time=-1) :
    """
    \Description : Apply a memetic algorithm to a population
    \Args : 
        popul       : the population of individual to evolve
        epochs      : number of epochs
        nb_best     : number of best individuals to select for crossover 
        pm          : probability for an individual to mutate
        radius      : the radius in which to search for the solution during the local search
        nb_neighb   : number of neighbours to test in the radius during the local search
        gen_max     : maximum nulber if generations
        inaccuracy  : desired inaccuracy. A value of -1 indicates that we do not want an exact value as objective
        time        : desired inference time. A value of -1 indicates that we do not want an exact value as objective
    \Outputs : None
    """
    # Pareto front of each generation
    pareto_frontiers = []
    
    # Measure starting time
    start = perf_counter()
    
    for gen in range(0, gen_max) :
        
        print("Generation {}".format(gen + 1))
        
        # --- EVALUATION ---
        eval_pop(popul, epochs)

        # --- LOCAL SEARCH ---
        local_search(popul, radius, nb_neighb)
        
        # Extract the current Pareto frontier and update the Pareto frontiers
        pareto_frontiers.append(pareto_front(popul))

        # Check if the objectives are met
        if(check_objectives(pareto_frontiers[gen], inaccuracy, time)) :
            break           # Exit the loop if the objectives are met

        # --- SELECTION --- 
        pop_next = selection(popul, nb_best)   
        
        # --- CROSSOVER ---
        pop_child = crossover(pop_next)

        # --- MUTATION ---
        mutation(pop_child, pm)
        
        # --- UPDATE OF THE CURRENT POPULATION ---  
        # The current population is the union of pop_next and pop_child's population
        popul.pop = pop_next.pop + pop_child.pop

    # end for gen
    
    # Measure ending time
    end = perf_counter()
    
    # Column names
    header = ["Pareto Frontier", "Inaccuracy (%)", "Time (s)", "NL", "NF", "lr", "mom", "Duration (s)"]  

    # Shaped data
    shaped_data = ut.shape_pareto_front(pareto_frontiers)
    
    # Add the duration at the end of the first line
    shaped_data[0].insert(len(shaped_data[1]), end-start)
    
    # Name of the file is the date and time
    filename = datetime.now().strftime("./results/mem_algo/%d-%m-%Y_%Hh%M.csv")
    
    # Write the data to the csv file
    ut.write_data_to_csv(filename, header, shaped_data)
    
    # Name of the file is the date and time
    filename = datetime.now().strftime("./results/mem_algo/%d-%m-%Y_%Hh%M.txt")

    # Save the best models' caracteristics
    for model in pareto_frontiers[len(pareto_frontiers) - 1] :
        model.printCNN(standard_out="file", filename=filename)
    
# end mem_algo()  
