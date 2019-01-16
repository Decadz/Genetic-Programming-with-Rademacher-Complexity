__author__ = "Christian Raymond"
__date__ = "07 December 2018"

"""
This is the primary script which is used to execute all the various implementations
of Genetic Programming for Symbolic Regression. To run this script make sure the 
global variables in main.py and config.py are set to the required settings. 
"""

import pandas as pd
import random as rd

from utility.exporter import output_to_file, default_columns, rademacher_columns
from algorithms import config


# Path to the data-set which is getting tested.
data_path_train = config.ccun_train
data_path_test = config.ccun_test

# Identification name of the data-set (appears in output file).
data_name = "ccun"


def main():

    # Loading the desired data-set.
    train_data = pd.read_csv(data_path_train, header=None)
    test_data = pd.read_csv(data_path_test, header=None)

    # Setting the training and testing data.
    config.training_data = train_data
    config.testing_data = test_data

    # Benchmark Genetic Programming Algorithms.
    run_benchmark_classic()

    # Experimental Genetic Programming Algorithms.
    run_experimental_rademacher_complexity()


"""
===============================================================================================================

    - Benchmark Genetic Programming Algorithms

===============================================================================================================
"""


def run_benchmark_classic():

    """
    A classic implementation of Genetic Programming for Symbolic Regression.
    This program aims to map the input data to the output data through the use
    of a symbolic representation (expression trees) and evolutionary techniques.
    """

    output_path = "../output/benchmark_classic/"

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = config.seeds.iloc[i].values[0]
        rd.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms.benchmark import benchmark_classic as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "benchmark-classic", i+1, data_name, statistics, default_columns)

        print("=== Benchmark Classic Execution " + str(i+1) + " Completed ===")


"""
===============================================================================================================

    - Experimental Genetic Programming Algorithms

===============================================================================================================
"""


def run_experimental_rademacher_complexity():

    """
    An experimental version of Genetic Programming which uses Rademacher Complexity
    to estimate the generalisation error. (Used as an alternative to VC dimensions).
    """

    output_path = "../output/experimental_rademacher_complexity/"

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = config.seeds.iloc[i].values[0]
        rd.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms.experimental import experimental_rademacher_complexity as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "experimental-rademacher-complexity", i+1, data_name,
                       statistics, rademacher_columns)

        print("=== Experimental Rademacher Complexity Execution " + str(i+1) + " Completed ===")


if __name__ == "__main__":
    main()
