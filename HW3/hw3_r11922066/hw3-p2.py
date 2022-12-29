import time
import random
import pandas as pd
import numpy as np
import copy
import math
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter 
from matplotlib import pyplot as plt


# ===========================================================
#                          Functions
# ===========================================================
score_history = []

def get_n_individual(counter, population):
    """
    If counter is 0, return the individual with the highest prob
    If counter is 1, return the second individual with the highest prob
    If counter is 2, return the third individual withthe highest prob
    """
    index = counter + 1
    probabilities = [ind[1] for ind in population]
    sorted_probs = sorted(probabilities, key=float)
    max_prob = probabilities[-index]
    max_individual = [ind[0] for ind in population if ind[1] == max_prob][0]
    
    return max_individual


def generate_random_individuals(num_individuals, num_features, max_features=None, verbose=False):
    """
    Randomly generates individuals

    The number of individuals to generate is given by the num_individuals parameter
    The length of each individual is equal to the num_features parameter
    The maximum number of active features for every individual is given by the max_features parameter
    """
    if verbose: print('GENERATING RANDOM INDIVIDUALS.... ')
        
    individuals = list()
    
    for _ in range(num_individuals):
        individual = ''
        for col in range(num_features):
            # For each char in the individual, a 1 or a 0 is randomly generated
            if individual.count('1') == max_features:
                individual += '0'
                continue
                
            individual += str(random.randint(0, 1))
            
        if verbose: print(f'Genrated a new indivudal: {individual}')
        individuals.append(individual)
        
    if verbose: print(f'Generated list of {num_individuals} individuals: {individuals}')
        
    return individuals


def get_weights(population):
    """
    Calculate weights from the population filled with the accuracies
    """
    total_accuracies = 0
    new_population = []
    
    # Get the sum of all accuracies of the population
    for individual in population:
        total_accuracies += individual[1]
        
    # For each individual, calculate its weight by dividing its accuracy by the overall sum calculated above
    for individual in population:
        weight = individual[1]/total_accuracies
        # Store the individual and its weight in the final population list
        new_population.append((individual[0], float(weight*100)))
        
    return new_population


def choose_parents(population, counter):
    """
    From the population, weighting the probabilities of an individual being chosen via the fitness
    function, takes randomly two individual to reproduce
    Population is a list of tuples, where the first element is the individual and the second
    one is the probability associated to it.
    To avoid generating repeated individuals, 'counter' parameter is used to pick parents in different ways, thus
    generating different individuals
    """
    # Pick random parent Number 1 and Number 2
    # (get_n_individual() function randomly picks an individual following the distribution of the weights)
    if counter == 0:        
        parent_1 = get_n_individual(0, population)        
        parent_2 = get_n_individual(1, population)
    elif counter == 1:
        parent_1 = get_n_individual(0, population)        
        parent_2 = get_n_individual(2, population)
        
    else:
        probabilities = (individual[1] for individual in population)
        individuals = [individual[0] for individual in population]
        parent_1, parent_2 = random.choices(individuals, weights=probabilities, k=2)
    
    return [parent_1, parent_2]


def mutate(child, prob=0.05):
    """
    Randomly mutates an individual according to the probability given by prob parameter
    """
    new_child = copy.deepcopy(child)
    for i, char in enumerate(new_child):
        if random.random() < prob:
            new_value = '1' if char == '0' else '0'
            new_child = new_child[:i] + new_value + new_child[i+1:]
    
    return new_child


def reproduce(individual_1, individual_2):
    """
    Takes 2 individuals, and combines their information based on a
    randomly chosen crosspoint.
    Each reproduction returns 2 new individuals
    """ 
    # Randomly generate a integer between 1 and the length of the individuals minus one, which will be the crosspoint
    crosspoint = random.randint(1, len(individual_1)-1)
    child_1 = individual_1[:crosspoint] + individual_2[crosspoint:]
    child_2 = individual_2[:crosspoint] + individual_1[crosspoint:]
    child_1, child_2 = mutate(child_1), mutate(child_2)
 
    return [child_1, child_2]


def generation_ahead(population, verbose=False):
    """
    Reproduces all the steps for choosing parents and making 
    childs, which means creating a new generation to iterate with
    """
    new_population = list()
    
    for _ in range(int(len(population)//2)):      
        # According to the weights calculated before, choose a set of parents to reproduce
        parents = choose_parents(population, counter=_)
        if verbose: print(f'Parents chosen: {parents}')
          
        # Reproduce the pair of individuals chose above to generate two new individuals
        childs = reproduce(parents[0], parents[1])
        if verbose: print(f'Generated children: {childs}\n')
        new_population += childs
        
    return new_population


def fill_population(individuals, x, y, verbose=False):
    """
    Fills the population list with individuals and their weights
    """
    population = list()
   
    for individual in individuals:

        # Get subset by individual
        x_subset = x[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
       
        # Build random forest
        #randomForest_clf = RandomForestClassifier(random_state=0)
        #randomForest_clf = SVC(kernel='rbf', random_state=0) 
        randomForest_clf = DecisionTreeClassifier(random_state=0)
        # Calculate validation score
        randomForest_scores = cross_val_score(randomForest_clf, x_subset, y, cv=5)

        # Get the value of the accuracy
        accuracy = randomForest_scores.mean()

        individual_complete = (individual, accuracy)
        score_history.append(individual_complete)
        population.append(individual_complete)
        
    # The final population list is created, which contains each individual together with its weight
    # (weights will be used in the reproduction step)
    new_population = get_weights(population)
    if verbose: print(f'Generated population list (with weights): {new_population}')
    
    return new_population
# ===========================================================
#                          Functions
# ===========================================================



# ===========================================================
#                          Load Data
# ===========================================================
# TODO: Load data here.
indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)
x = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T
y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
y = (y>0).astype(int).reshape(y.shape[0])
# ===========================================================
#                          Load Data
# ===========================================================


# ===========================================================
#              Genetic Algorithm Feature Selection
# =========================================================== 
def main_loop(ind_num, x, y, max_iter=5, verbose=False):
    """
    Performs all the steps of the Genetic Algorithm
    1. Generate random population
    2. Fill population with the weights of each individual
    3. Check if the goal state is reached
    4. Reproduce the population, and create a new generation
    5. Repeat process until termination condition is met
    """
    # Generate individuals (returns a list of strings, where each str represents an individual)
    individuals = generate_random_individuals(ind_num, x.shape[1], 200)
    # Returns a list of tuples, where each tuple represents an individual and its weight
    population = fill_population(individuals, x, y)

    # Reproduce current generation to generate a better new one
    new_generation = generation_ahead(population, verbose)
    
    # After the new generation is generated, the loop goes on until a solution is found or until the maximum number of
    # iterations are reached
    iteration_count = 0
    while iteration_count < max_iter:
        if verbose: print(f'\n\n\nITERATION NUMBER {iteration_count+1} (Iteration max = {max_iter+1})\n\n\n')
        population = fill_population(new_generation, x, y)
        
        new_generation = generation_ahead(population, verbose)   
        iteration_count += 1
        
    return population


main_loop(200, x, y, 5)
max_score = max(score_history,key=itemgetter(1))
print(f"Max: {max_score[1]}")
print(f"Number of features: {max_score[0].count('1')}")
print("Selected features: ")
for i in range(len(max_score[0])):
    if(max_score[0][i] == '1'):
        print( str(indexes[0][i])+',', end=' ' )
print("\n")
# ===========================================================
#              Genetic Algorithm Feature Selection
# ===========================================================


# ===========================================================
#                       Visualization
# =========================================================== 
plt.plot(range(len(score_history)), [t[1] for t in score_history])
plt.title('Genetic Algorithm Feature Selection')
plt.xlabel('id of individual')
plt.ylabel('Cross-validation score')
plt.savefig('2-3-6_result.png')
# ===========================================================
#                       Visualization
# =========================================================== 