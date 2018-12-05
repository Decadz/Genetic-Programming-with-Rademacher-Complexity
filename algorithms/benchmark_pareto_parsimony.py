__author__ = "Christian Raymond"
__date__ = "05 December 2018"

"""
A multi-objective (pareto) implementation of Genetic Programming for Symbolic
Regression. This program aims to optimise for both fitness and parsimony (size
of expression tree). This is used as a benchmark for parsimony pressure.
"""

import numpy as np
import operator

from utility.operators import division
from utility.evaluate import evaluate_population

from deap import base
from deap import creator
from deap import tools
from deap import gp


def main(training_data, testing_data, seed, toolbox, prob_crossover, prob_mutation, num_generations,
         size_population, random_upper, random_lower):

    """
    Pareto Parsimony Pressure Genetic Programming for Symbolic Regression, initialises
    and executes (trains and tests) the algorithm.

    :param training_data: Training Data
    :param testing_data: Testing Data
    :param seed: Random seed
    :param toolbox: Evolutionary operators
    :param prob_crossover: Crossover rate
    :param prob_mutation: Mutation rate
    :param num_generations: Number of Generations
    :param size_population: Population Size
    :param random_upper: Random Terminal Max
    :param random_lower: Random Terminal Min
    :return statistics: List of Statistics
    :return population: Final Population
    :return halloffame: Best Individuals
    """

    # Setting the seed (for reproducibility).
    np.random.seed(seed)

    # Create and set up genetic program.
    initialise_algorithm(training_data, toolbox, random_upper, random_lower)

    return execute_algorithm(testing_data, toolbox, prob_crossover, prob_mutation, num_generations, size_population)


def initialise_algorithm(training_data, toolbox, random_upper, random_lower):

    """
    Sets up the hyper parameters for the algorithm, adding all the
    genetic operators and terminals etc.

    :param training_data: Training Data
    :param toolbox: Evolutionary Operators
    :param random_upper: Random Terminal Max
    :param random_lower: Random Terminal Min
    """

    # Creating a new primitive node set.
    pset = gp.PrimitiveSet("main", training_data.shape[1] - 1)

    # Adding operators to the primitive set.
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(division, 2)

    # Adding some random terminals between a set range with a set #dp.
    pset.addEphemeralConstant("randomTerm", lambda: round(np.random.uniform(random_lower, random_upper), 4))

    # Tell the algorithm that we are trying to minimise the fitness function.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Need to use either SPEA2 (Strength Pareto Evolutionary Algorithm) or
    # NSGA2 (Non-dominated Sorting Genetic Algorithm) for multi-objective.
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", fitness_function_ae, data=training_data, toolbox=toolbox)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Restricting size of expression tree to avoid bloat.
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=50))


def execute_algorithm(testing_data, toolbox, prob_crossover, prob_mutation, num_generations, size_population):

    """
    Sets up the statistics and output of the genetic program, before executing
    the genetic program output of program visible in console.

    :param testing_data: Testing Data
    :param toolbox: Evolutionary Operators
    :param prob_crossover: Crossover rate
    :param prob_mutation: Mutation Rate
    :param num_generations: Number of Generations
    :param size_population: Size of Population
    :return statistics: List of Statistics
    :return population: Final Population
    :return halloffame: Best Individuals
    """

    population = toolbox.population(n=size_population)
    halloffame = tools.ParetoFront()
    statistics = []

    # Begin the evolution process.
    for generation in range(num_generations):
        print("Generation %i" % generation)

        # Evaluate the populations fitness.
        for individual, fitness in zip(population, list(map(toolbox.evaluate, population))):
            individual.fitness.values = fitness

        # Update the hall of fame.
        halloffame.update(population)

        # Gather all the fitnesses and size information about the population.
        fits = [individual.fitness.values[0] for individual in population]
        size = [individual.fitness.values[1] for individual in population]

        # Evaluates the population and returns statistics.
        statistics.append(evaluate_population(generation, fits, size, halloffame[0],
                                    toolbox.compile(expr=halloffame[0]), testing_data))

        # Select the next generation of individuals and clone.
        next_gen = toolbox.select(population, size_population)
        next_gen = list(map(toolbox.clone, next_gen))

        # Apply crossover genetic operators.
        for i in range(1, size_population, 2):
            if np.random.random() < prob_crossover:
                next_gen[i - 1], next_gen[i] = toolbox.mate(next_gen[i - 1], next_gen[i])
                del next_gen[i - 1].fitness.values
                del next_gen[i].fitness.values

        # Apply mutation genetic operators.
        for mutant in next_gen:
            if np.random.random() < prob_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # The population is entirely replaced by the offspring
        population[:] = next_gen

    return statistics, population, halloffame


"""
===============================================================================================================

    - Fitness Functions

===============================================================================================================
"""


def fitness_function_ae(individual, data, toolbox):

    """
    Calculates the fitness of a candidate solution/individual by using the absolute
    value of the errors (AE).

    :param individual: Candidate Solution
    :param data: Evaluation Data
    :param toolbox: Evolutionary Operators
    :return: Fitness Value (Error)
    :return: Size of Individual
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1] - 1]))
        real = data.values[rows][data.shape[1] - 1]

        error = abs(real - pred)
        total_error += error

    # Must return the value as a list object.
    return total_error, len(individual)


def fitness_function_sse(individual, data, toolbox):

    """
    Calculates the fitness of a candidate solution/individual by using the sum of
    the squared errors (SSE).

    :param individual: Candidate Solution
    :param data: Evaluation Data
    :param toolbox: Evolutionary Operators
    :return: Fitness Value (Error)
    :return: Size of Individual
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1] - 1]))
        real = data.values[rows][data.shape[1] - 1]

        error = (real - pred) ** 2
        total_error += error

    # Must return the value as a list object.
    return total_error, len(individual)


def fitness_function_mse(individual, data, toolbox):

    """
    Calculates the fitness of a candidate solution/individual by using the mean
    of the squared errors (MSE).

    :param individual: Candidate Solution
    :param data: Evaluation Data
    :param toolbox: Evolutionary Operators
    :return: Fitness Value (Error)
    :return: Size of Individual
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1] - 1]))
        real = data.values[rows][data.shape[1] - 1]

        error = (real - pred) ** 2
        total_error += error

    # Must return the value as a list object.
    return total_error/data.shape[0], len(individual)
