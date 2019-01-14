__author__ = "Christian Raymond"
__date__ = "12 January 2019"

"""
An experimental version of Genetic Programming for Symbolic Regression which uses 
Rademacher Complexity to estimate the generalisation error.
"""

import algorithms.config as config
import random as rd
import numpy as np

import operator
import time

from utility.operators import division
from utility.evaluate import evaluate_population_rademacher

from deap import base
from deap import creator
from deap import tools
from deap import gp


error_theta = 0.5
complexity_theta = 0.5

# Random rademacher vector which contains [1, -1] values.
random_rademacher_vector = list()


def execute_algorithm():

    """
    Executes the genetic program, evolving candidate solutions using
    the parameters specified in the config.py file.

    :return statistics: List of Statistics
    :return population: Final Population
    :return halloffame: Best Individuals
    """

    population = toolbox.population(n=config.size_population)
    halloffame = tools.HallOfFame(config.size_population * config.prob_elitism)
    statistics = []

    # Begin the evolution process.
    for generation in range(config.num_generations):

        print("Generation: " + str(generation + 1) + "/" + str(config.num_generations))
        start_time = time.clock()  # Records the time at the start of the generation.

        # Reseting the random vector list so it is empty before populating.
        random_rademacher_vector.clear()

        # Generate a random_vector containing Rademacher random variables (+1, -1).
        for i in range(20):
            random_rademacher_vector.append([rd.randint(0, 1) * 2 - 1 for x in range(len(config.training_data))])

        """
        ====================================================================
        The new evaluation method below. (Note this is not important to the 
        concept, just a work around required due to the DEAP implementation).
        Records all the errors and complexities then noramlises the errors
        before adding them to the complexities.
        ====================================================================
        """

        residuals = []
        complexities = []
        fitnesses = []

        # Evaluate the populations fitness.
        for individual, fitness in zip(population, list(map(toolbox.evaluate, population))):
            residuals.append(fitness[0])
            complexities.append(fitness[1])

        # Calculating the range of the residuals/errors so we can normalise.
        error_max = max(residuals)
        error_min = min(residuals)
        error_range = error_max - error_min

        for index, individual in enumerate(population):

            # Calculating the individuals fitness.
            ind_error = error_theta*(residuals[index] / error_range)
            ind_complexity = complexity_theta*(complexities[index])
            ind_fitness = ind_error + ind_complexity

            # Calculating and setting the new fitness value.
            fitnesses.append(ind_fitness)
            individual.fitness.values = [fitnesses[index]]

        """ 
        ====================================================================
        End of new evaluation method - the rest of this function is same as
        the classic implementation of genetic programming. 
        ====================================================================
        """

        # Update the hall of fame.
        halloffame.update(population)

        # Gather all the size information about the population.
        size = [len(individual) for individual in population]

        # Cloning the population before performing genetic operators.
        next_gen = toolbox.select(population, config.size_population + 1)
        next_gen = list(map(toolbox.clone, next_gen))

        children = []

        # Performing elitism genetic operator.
        for index in range(len(halloffame)):
            children.append(halloffame[index])

        # Populating the rest of the next generation with crossover and mutation.
        while len(children) < config.size_population:

            # Generating a number to determine what genetic operator.
            rng = rd.random()

            # Performing crossover genetic operator.
            if rng < config.prob_crossover:
                index1 = len(children)
                index2 = len(children) + 1
                next_gen[index1], next_gen[index2] = toolbox.mate(next_gen[index1], next_gen[index2])
                del next_gen[index1].fitness.values
                del next_gen[index2].fitness.values
                children.extend((next_gen[index1], next_gen[index2]))

            # Performing mutation genetic operator.
            elif rng < config.prob_crossover + config.prob_mutation:
                index = len(children)
                toolbox.mutate(next_gen[index])
                del next_gen[index].fitness.values
                children.append(next_gen[index])

        population = children  # The population is entirely replaced by the offspring
        end_time = time.clock()  # Records the time at the end of the generation.

        # Evaluates the population and returns statistics.
        statistics.append(evaluate_population_rademacher(generation, end_time - start_time, fitnesses, residuals,
                                                         complexities, size, halloffame[0],
                                                         toolbox.compile(expr=halloffame[0])))

    return statistics, population, halloffame


"""
===============================================================================================================

    - Fitness Functions

===============================================================================================================
"""


def fitness_function_mse(individual, data, toolbox):

    """
    Calculates the fitness of a candidate solution/individual by using the mean
    of the squared errors (MSE). Function also returns the rademacher complexity
    as a 2nd argument.

    :param individual: Candidate Solution
    :param data: Evaluation Data
    :param toolbox: Evolutionary Operators
    :return: mean squared error [0], rademacher complexity [1]
    """

    """
    ====================================================================
    Calculating the MSE as per a typical fitness function in GP, the 
    only difference is that we want to record the predictions for use
    later in a hypothesis vector (since we don't want to recalculate it).
    ====================================================================
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)

    # A list of predictions made by the individual on the training data.
    hypothesis_vector = []

    # The total (MSE) error made by the individual.
    total_error = 0

    for rows in range(data.shape[0]):
        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1] - 1]))
        real = data.values[rows][data.shape[1] - 1]
        hypothesis_vector.append(pred)

        # Updating the total error made by the individual.
        error = (real - pred) ** 2
        total_error += error

    """
    ====================================================================
    The rademacher complexity is used in a binary classification, so we
    need to convert the output range of the hypothesis to [0, 1], 
    currently we have a continuous value output since we are performing
    (symbolic) regression. 
    ====================================================================
    """

    output_range = max(hypothesis_vector) - min(hypothesis_vector)

    if output_range != 0:
        # Normalising the hypothesis output to a range of [1, -1].
        for p in range(len(hypothesis_vector)):
            hypothesis_vector[p] = hypothesis_vector[p] / output_range
            if hypothesis_vector[p] >= 0: hypothesis_vector[p] = 1
            if hypothesis_vector[p] < 0: hypothesis_vector[p] = -1

    elif output_range == 0:
        # All outputs set to 1, since the vector has 0 range. This is
        # called when on functions such as f = argXi - where there is
        # no range, just a constant output.
        for p in range(len(hypothesis_vector)):
            hypothesis_vector[p] = 1

    """
    ====================================================================
    This is where the calculation of the rademacher complexity happens.
    A random rademacher vector "r" the same length as the training set is
    generated, which contains [+1, -1] values. This vector is compared to
    the hypothesis vector "h" recorded prior. 

    When the value of ri and hi agree the resulting correlation value is
    0 when they disagree the correlation value is 1. 
        i.e.    (1, 1) & (-1, -1) = 0
                (1, -1) & (-1, 1) = 1
    ====================================================================
    """

    # Finding the length of the training set.
    m = len(config.training_data)

    complexity = []
    num_samples = 20

    for s in range(num_samples):

        random_vector = random_rademacher_vector[s]

        correlations = []
        for i in range(m):
            correlation = 0 if hypothesis_vector[i] == random_vector[i] else 1
            correlations.append(correlation)

        complexity.append(sum(correlations))

    # Calculating the rademacher complexity and multiply it by 2 to
    # transform its range from [0, 0.5] -> [0, 1].
    hypothesis_complexity = 2 * (0.5 - (1 / (2 * m)) * (sum(complexity) / num_samples))

    """
    ====================================================================
    Returning both the mse and the rademacher complexity, note that we 
    do not add the training error to the rademacher complexity yet since
    we need to normalise the populations errors first to [0, 1] which 
    cannot be done in the fitness function.
    ====================================================================
    """

    # Mean squared error - must return the error value as a list object.
    mse = [total_error / data.shape[0]][0]

    # The rademacher complexity of this current hypothesis/candidate solution.
    rademacher_complexity = hypothesis_complexity

    return [mse, rademacher_complexity]


"""
===============================================================================================================

    - Algorithm Configuration (this portion of code must remain in global scope as per DEAP requirements).

===============================================================================================================
"""

# Creating a new primitive node set.
pset = gp.PrimitiveSet("main", config.training_data.shape[1] - 1)

# Adding operators to the primitive set.
pset.addPrimitive(np.add, 2, name="add")
pset.addPrimitive(np.subtract, 2, name="sub")
pset.addPrimitive(np.multiply, 2, name="mul")
pset.addPrimitive(division, 2, name="div")

# Adding some random terminals between a set range with a set #dp.
pset.addEphemeralConstant("randomTerm2", lambda:
round(rd.randint(config.random_lower * 10000, config.random_upper * 10000) / 10000, 4))

# Tell the algorithm that we are trying to minimise the fitness function.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# Creating a toolbox of evolutionary operators.
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", fitness_function_mse, data=config.training_data, toolbox=toolbox)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Restricting size of expression tree to avoid bloat.
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=config.max_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=config.max_height))
