__author__ = "Christian Raymond"
__date__ = "06 December 2018"

"""
This is the primary script which is used to execute all the various implementations
of Genetic Programming for Symbolic Regression. To run this script make sure the 
global variables in main.py and config.py are set to the required settings. 
"""

import pandas as pd
import numpy as np

from utility.exporter import output_to_file
from algorithms import config


# Path to the data-set which is getting tested.
data_path_train = config.experimental_train_1
data_path_test = config.experimental_test_1

# Identification number of the data-set (appears in output file).
data_name = "experimental"


def main():

    # TODO - Fix error which doesnt allow running different gp scripts one after the other.
    # TODO - Implement README's in all the desired locations.
    # TODO - Get DEAP working at full speed, currently going back a py version.
    # TODO - Delete experimental files when program all works.

    run_benchmark_classic()
    run_benchmark_pareto_parsimony()


def run_benchmark_classic():

    """
    A multi-objective (pareto) implementation of Genetic Programming for Symbolic
    Regression. This program aims to optimise for both fitness and parsimony (size
    of expression tree). This is used as a benchmark for parsimony pressure.
    """

    output_path = "../output/benchmark_classic/"

    # Loading the desired data-set.
    train_data = pd.read_csv(data_path_train, header=None)
    test_data = pd.read_csv(data_path_test, header=None)

    # Setting the training and testing data.
    config.training_data = train_data
    config.testing_data = test_data

    # Loading the seed values from file
    seeds = pd.read_csv(config.seed, header=None)

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = seeds.iloc[i]
        np.random.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms import benchmark_classic as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "benchmark-classic", i+1, data_name, statistics)

        print("=== Benchmark Classic Execution " + str(i+1) + " Completed ===")


def run_benchmark_pareto_parsimony():

    """
    A classic implementation of Genetic Programming for Symbolic Regression.
    This program aims to map the input data to the output data through the use
    of a symbolic representation (expression trees) and evolutionary techniques.
    """

    output_path = "../output/benchmark_pareto_parsimony/"

    # Loading the desired data-set.
    train_data = pd.read_csv(data_path_train, header=None)
    test_data = pd.read_csv(data_path_test, header=None)

    # Setting the training and testing data.
    config.training_data = train_data
    config.testing_data = test_data

    # Loading the seed values from file
    seeds = pd.read_csv(config.seed, header=None)

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = seeds.iloc[i]
        np.random.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms import benchmark_pareto_parsimony as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "benchmark-pareto-parsimony", i+1, data_name, statistics)

        print("=== Benchmark Pareto Parsimony Execution " + str(i+1) + " Completed ===")


if __name__ == "__main__":
    main()
