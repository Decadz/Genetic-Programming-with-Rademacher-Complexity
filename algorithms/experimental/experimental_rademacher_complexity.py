__author__ = "Christian Raymond"
__date__ = "14 December 2018"

"""
An experimental version of Genetic Programming for Symbolic Regression which uses 
Rademacher Complexity to estimate the generalisation error.
"""

import algorithms.config as config
import numpy as np
import operator
import time

from utility.operators import division
from utility.evaluate import evaluate_population

from deap import base
from deap import creator
from deap import tools
from deap import gp


# Samples used to estimate the Rademacher complexity.
num_estimate_samples = 500


def execute_algorithm():

    """
    Executes the genetic program, evolving candidate solutions using
    the parameters specified in the config.py file.

    :return statistics: List of Statistics
    :return population: Final Population
    :return halloffame: Best Individuals
    """

    population = toolbox.population(n=config.size_population)
    halloffame = tools.HallOfFame(config.size_population*config.prob_elitism)
    statistics = []

    # Begin the evolution process.
    for generation in range(config.num_generations):

        print("Generation: " + str(generation + 1) + "/" + str(config.num_generations))
        start_time = time.clock()  # Records the time at the start of the generation.

        # Evaluate the populations fitness.
        for individual, fitness in zip(population, list(map(toolbox.evaluate, population))):
            individual.fitness.values = fitness

        # Update the hall of fame.
        halloffame.update(population)

        # Gather all the fitness's and size information about the population.
        fits = [individual.fitness.values[0] for individual in population]
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
            rng = np.random.random()

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
        statistics.append(evaluate_population(generation, end_time - start_time, fits, size, halloffame[0],
                                              toolbox.compile(expr=halloffame[0])))

    return statistics, population, halloffame


"""
===============================================================================================================

    - Rademacher Complexity

===============================================================================================================
"""


def rademacher_estimate(training_vector, classifier):

    """
    Given a (training) data-set, estimate the upper genralisation bounds using emperical Rademacher Complexity.
    This is performed over a number of estimations taking the average, which will give close to the expected
    value due to the law of large numbers (LLN).

    :param training_vector: a vector of training data (pre-defined).
    :param classifier: a function that generates an iterator over hypotheses given a data-set
    """

    totals = list()  # List of upper bounds from each iteration.

    for i in range(num_estimate_samples):  # Iterating multiple times to get a good approximation.

        # Generate a tuple of random_vector - random variables (+1, -1)
        random_vector = rademacher_random_variable(len(training_vector))

        # Finding the Rademacher correlations/bounds on the random_vector.
        bounds = [h.correlation(training_vector, random_vector) for h in list(classifier(training_vector))]

        # Adding the upper bound to the totals list.
        totals.append(max(bounds))

        print("==========================================================================")
        print("Bounds:", bounds)
        print("Upper Bounds:", max(bounds))
        print("Iteration:", i)
        print("==========================================================================")
        print()

    return sum(totals)/num_estimate_samples  # Averaging all the upper bounds to come up with the best estimate.


def rademacher_random_variable(size):

    """
    Generate a desired number of rademacher_random_variable (with values {+1, -1})

    :param size: The number of Rademacher random variables to generate.
    :return tuple: Tuple full of rademacher random variables.
    """

    return [np.random.randint(0, 1) * 2 - 1 for x in range(size)]


def rademacher_correlation(self, training_vector, random_vector):

    """
    Return the rademacher correlation between a label assignment and the predictions of
    the classifier

    :param hypothesis_vector: A vector of the training data
    :param random_vector: A vector of Rademacher random variables (+1/-1)
    """

    hypothesis_vector = [1 if (self.classify(d)) else -1 for d in training_vector]

    # Product of random_vector
    dot_product = float(np.dot(hypothesis_vector, random_vector))

    # Number of features we are testing.
    size = float(len(training_vector))

    # Divide by size to push the number between 1 and -1.
    correlation = dot_product / size

    print("Training Vector:", training_vector)
    print("Rademacher Vector:", random_vector)
    print("Hypothesis Vector:", hypothesis_vector)
    print("Dot Product:", dot_product)
    print("Correlation:", correlation)
    print()

    return correlation


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
    :return: Fitness Value
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
    return [total_error]


def fitness_function_sse(individual, data, toolbox):

    """
    Calculates the fitness of a candidate solution/individual by using the sum of
    the squared errors (SSE).

    :param individual: Candidate Solution
    :param data: Evaluation Data
    :param toolbox: Evolutionary Operators
    :return: Fitness Value
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
    return [total_error]


def fitness_function_mse(individual, data, toolbox):

    """
    Calculates the fitness of a candidate solution/individual by using the mean
    of the squared errors (MSE).

    :param individual: Candidate Solution
    :param data: Evaluation Data
    :param toolbox: Evolutionary Operators
    :return: Fitness Value
    """

    # Converts the expression tree into a callable function.
    func = toolbox.compile(expr=individual)
    total_error = 0

    upper_range = float("inf")
    lower_range = float("-inf")

    for rows in range(data.shape[0]):

        # Uses splat operator to convert array into positional arg.
        pred = func(*(data.values[rows][0:data.shape[1]-1]))
        real = data.values[rows][data.shape[1]-1]

        error = (real - pred)**2
        total_error += error

    # Must return the value as a list object.
    mse = [total_error/data.shape[0]]

    # Need to get the range of the output on training and convert to binary [-1, 1] by dividing output by range.
    range = upper_range - lower_range

    return mse


"""
===============================================================================================================

    - Algorithm Configuration (this portion of code must remain in global scope as per DEAP requirements).

===============================================================================================================
"""

# Creating a new primitive node set.
pset = gp.PrimitiveSet("main", config.training_data.shape[1] - 1)

# Adding operators to the primitive set.
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(division, 2)

# Adding some random terminals between a set range with a set #dp.
pset.addEphemeralConstant("randomTerm5", lambda: round(np.random.uniform(config.random_lower, config.random_upper), 4))

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
