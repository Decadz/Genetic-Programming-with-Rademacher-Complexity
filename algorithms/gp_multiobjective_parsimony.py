__author__ = "Christian Raymond"
__date__ = "29 November 2018"


import numpy as np
import random as rd
import operator

from utility.loader import load_data
from utility.operators import division

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# Setting the seed (for reproducibility).
np.random.seed(2018)
rd.seed(2018)

# Genetic Operators.
prob_crossover = 0.7
prob_mutation = 0.1

# Population and Generations.
num_generations = 300
size_population = 300

# Random terminal limits.
random_upper = 5
random_lower = 1

# Path to regression data-sets.
data_1 = "../data/regression-1.csv"
data_2 = "../data/regression-2.csv"

data = load_data(data_1)
toolbox = base.Toolbox()


def main():

    """
    Pareto Parsimony Pressure Genetic Programming for Symbolic Regression, initialises
    and executes the algorithm. (Ensure load_data() points to the right file path).
    """

    # Create and set up genetic program.
    initialise_algorithm()

    # Execute the genetic program.
    execute_algorithm()


def initialise_algorithm():

    """
    Sets up the hyper parameters for the algorithm, adding all the
    genetic operators and terminals etc.
    """

    # Creating a new primitive node set.
    pset = gp.PrimitiveSet("main", data.shape[1]-1)

    # Adding operators to the primitive set.
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(division, 2)

    # Adding some random terminals between a set range with a set #dp.
    pset.addEphemeralConstant("randomTerm", lambda: round(rd.uniform(random_lower, random_upper), 4))

    # Tell the algorithm that we are trying to minimise the fitness function.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Need to use either SPEA (Strength Pareto Evolutionary Algorithm) or
    # NSGA (Non-dominated Sorting Genetic Algorithm) for multi-objective.
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", fitness_function_ae)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Restricting size of expression tree to avoid bloat.
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


def execute_algorithm():

    """
    Sets up the statistics and output of the genetic program, before executing
    the genetic program output of program visible in console.
    """

    population = toolbox.population(n=size_population)
    halloffame = tools.ParetoFront()

    # What stat are going to be seen in the console.
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)

    # What metrics are going to be display to the console.
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("max", np.max, axis=0)
    mstats.register("mean", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)

    # Run the genetic programming algorithm.
    population, statistics = algorithms.eaSimple(population, toolbox, cxpb=prob_crossover, mutpb=prob_mutation,
                                  ngen=num_generations, stats=mstats, halloffame=halloffame, verbose=True)

    # Display the information about the best solution.
    print("Best Solution:", halloffame[0])
    print("Solution Fitness:", halloffame[0].fitness)
    print("Solution Size:", len(halloffame[0]))


"""
===============================================================================================================

    - Fitness Functions

===============================================================================================================
"""


def fitness_function_ae(individual):

    """
    Calculates the fitness of a candidate solution/individual by using the absolute
    value of the errors (AE).

    :param individual: Candidate Solution
    :return: Fitness Value (Error)
    :return: Size of Individual
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1]-1]))
        real = data.values[rows][data.shape[1]-1]

        error = abs(real - pred)
        total_error += error

    # Must return the value as a list object.
    return total_error, len(individual)


def fitness_function_sse(individual):

    """
    Calculates the fitness of a candidate solution/individual by using the sum of
    the squared errors (SSE).

    :param individual: Candidate Solution
    :return: Fitness Value (Error)
    :return: Size of Individual
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1]-1]))
        real = data.values[rows][data.shape[1]-1]

        error = (real - pred)**2
        total_error += error

    # Must return the value as a list object.
    return total_error, len(individual)


def fitness_function_mse(individual):

    """
    Calculates the fitness of a candidate solution/individual by using the mean
    of the squared errors (MSE).

    :param individual: Candidate Solution
    :return: Fitness Value (Error)
    :return: Size of Individual
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1]-1]))
        real = data.values[rows][data.shape[1]-1]

        error = (real - pred)**2
        total_error += error

    # Must return the value as a list object.
    return total_error/data.shape[0], len(individual)


if __name__ == "__main__":
    main()
