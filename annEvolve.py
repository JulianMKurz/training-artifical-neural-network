# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:12:01 2017

@author: Ridvan Aydin Sibic
@author: Julian Kurz
"""
import numpy
import random
import copy
import math
import matplotlib.pyplot as plt

sigmoid = lambda x: 1 / (1 + numpy.exp(-x))
cosine = lambda x: numpy.cos(x)
tanh_activiation = lambda x: numpy.tanh(x)


linear_func = lambda x: x
quadratic_func = lambda x: x**2
cubic_func = lambda x: x**3
sin_func = lambda x: math.sin(x)
tanh_func = lambda x: math.tanh(x)

class ArtificialNeuralNetwork(object):
    def __init__(self, topology, activitationFunction):
        self.activitationFunction = activitationFunction
        self.topology = topology
        self.biases = []
        self.weights = []
        for bias in topology[1:]:
            self.biases.append(numpy.random.randn(bias,1)*5/3)
        for i, j in zip(topology[:-1],topology[1:]):
            self.weights.append(numpy.random.randn(j, i)*5/3)
            
    def output(self, input_value):
        for bias, weight in zip(self.biases, self.weights):
            input_value = self.activitationFunction(numpy.dot(weight, input_value)+bias)
            
        return input_value

def create_pop(size, topology, func, activiationFunction):
    population_neural_network = []
    for i in range(size):
        ann = ArtificialNeuralNetwork(topology, activiationFunction)
        member = [ann,fitness_function(ann, func)]
        population_neural_network.append(member)
        
    return population_neural_network

def create_pop_with_random_topology(size, func, activiationFunction):
    population_neural_network = []
    for i in range(size):
        layers = random.randint(0, 5)
        topology = [1]
        for l in range(layers):
            nodes = random.randint(1, 10)
            topology.append(nodes)
        topology.append(1)
        ann = ArtificialNeuralNetwork(topology, activiationFunction)
        print(topology)
        member = [ann,fitness_function(ann, func)]
        population_neural_network.append(member)
        
    return population_neural_network

def fitness_function(ann, func):
    error = 0.0
    for i in range(10):
        random_input = i/10
        #random_input = i
        err = (func(random_input) - ann.output(random_input))**2
        error += err
    return error/10

def fittest_in_pop(pop):
    fittest_member = pop[0]
    for member in pop:
        if(member[1]<fittest_member[1]):
            fittest_member = member
    return fittest_member

def remove_weakest(pop):
    weakest_member = pop[0]
    for member in pop:
        if(member[1]>weakest_member[1]):
            weakest_member = member
    pop.remove(weakest_member)

def tournament_selection(pop,size):
    tournament = []
    for i in range(size):
        random_index = random.randint(0, len(pop)-1)
        tournament.append(pop[random_index])
        
    return fittest_in_pop(tournament)

def mutate(parent, func):
    child = copy.deepcopy(parent)
    mutation_factor = random.uniform(-1.0,1.0)
    
    if random.random() >= 0.5:
        weights = child[0].weights
        random_layer = random.randint(0, len(weights)-1)
        random_node = random.randint(0, len(weights[random_layer])-1)
        random_weight = random.randint(0, len(weights[random_layer][random_node])-1)
        
        weights[random_layer][random_node][random_weight] += mutation_factor
        
    else:
        biases = child[0].biases
        random_layer = random.randint(0, len(biases)-1)
        random_node = random.randint(0, len(biases[random_layer])-1)
        
        biases[random_layer][random_node][0] += mutation_factor
        
    child[1] = fitness_function(child[0], func)
    return child

def mutate_activation_func(parent, func):
    child = copy.deepcopy(parent)
    randomNumber = random.randint(1, 3)
    
    if randomNumber == 1:
        new_activation_func = sigmoid
    if randomNumber == 2:
        new_activation_func = tanh_activiation
    if randomNumber == 3:
        new_activation_func = cosine
    
    child[0].activitationFunction =  new_activation_func
        
    child[1] = fitness_function(child[0], func)
    return child

"""
Sample experiment, shown below.

"""

population = create_pop_with_random_topology(10,quadratic_func,tanh_activiation)

resultChartX = []
resultChartY = []

for i in range(100000):
    if i % 100 == 0:
        fittest = fittest_in_pop(population)
        resultChartX.append(i)
        resultChartY.append(fittest[1][0][0])
    Parent = tournament_selection(population, 50)
    
    if random.random() >= 0.2:
        Child = mutate(Parent, quadratic_func)
    else:
        Child = mutate_activation_func(Parent, quadratic_func)
    remove_weakest(population)
    population.append(Child)
    
solution = fittest_in_pop(population)

print(solution[0].output(0.1))
print(solution[0].output(0.5))
print(solution[0].output(0.9))

print(solution[1])
print(solution[0].topology)
print(solution[0].weights)
print(solution[0].biases)
    
plt.plot(resultChartX,resultChartY)
plt.ylabel("Fitness")
plt.xlabel("Number of Generations")
plt.show()
    
    
    
    
    
    
    
    