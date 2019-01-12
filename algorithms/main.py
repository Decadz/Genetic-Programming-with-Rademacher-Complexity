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
data_path_train = config.experimental_train_1
data_path_test = config.experimental_test_1

# Identification name of the data-set (appears in output file).
data_name = "experimental-v2"


def main():

    # Loading the desired data-set.
    train_data = pd.read_csv(data_path_train, header=None)
    test_data = pd.read_csv(data_path_test, header=None)

    # Setting the training and testing data.
    config.training_data = train_data
    config.testing_data = test_data

    # Benchmark Genetic Programming Algorithms.
    #run_benchmark_classic()
    #run_benchmark_pareto_parsimony()
    #run_benchmark_diversity()
    #run_benchmark_spea2_parsimony()

    # Experimental Genetic Programming Algorithms.
    #run_experimental_rademacher_complexity()
    run_experimental_rademacher_complexity_v2()


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


def run_benchmark_pareto_parsimony():

    """
    A multi-objective (pareto) implementation of Genetic Programming for Symbolic
    Regression. This program aims to optimise for both fitness and parsimony (size
    of expression tree). This is used as a benchmark for parsimony pressure.
    """

    output_path = "../output/benchmark_pareto_parsimony/"

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = config.seeds.iloc[i].values[0]
        rd.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms.benchmark import benchmark_pareto_parsimony as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "benchmark-pareto-parsimony", i+1, data_name, statistics)

        print("=== Benchmark Pareto Parsimony Execution " + str(i+1) + " Completed ===")


def run_benchmark_diversity():

    """
    A diversity maintaining implementation of Genetic Programming for Symbolic
    Regression. This program aims to manage the diversity of individuals by using
    a domination based selection scheme. "Multi-Objective Methods for Tree Size
    Control" by Edwin D. de Jong, Jordan B. Pollack.
    """

    output_path = "../output/benchmark_diversity/"

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = config.seeds.iloc[i].values[0]
        rd.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms.benchmark import benchmark_diversity as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "benchmark-diversity", i+1, data_name, statistics)

        print("=== Benchmark Diversity Execution " + str(i+1) + " Completed ===")


def run_benchmark_spea2_parsimony():

    """
    A multi-objective implementation of Genetic Programming for Symbolic Regression.
    This program uses the Strength Pareto Evolutionary Algorithm (SPEA2) selection.
    """

    output_path = "../output/benchmark_spea2_parsimony/"

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = config.seeds.iloc[i].values[0]
        rd.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms.benchmark import benchmark_spea2_parsimony as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "benchmark-spea2-parsimony", i+1, data_name, statistics)

        print("=== Benchmark SPEA2 Parsimony Execution " + str(i+1) + " Completed ===")


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


def run_experimental_rademacher_complexity_v2():

    """
    An experimental version of Genetic Programming which uses Rademacher Complexity
    to estimate the generalisation error. (Used as an alternative to VC dimensions).
    """

    output_path = "../output/experimental_rademacher_complexity_v2/"

    for i in range(config.executions):

        # Setting the seed (for reproducibility).
        config.cur_seed = config.seeds.iloc[i].values[0]
        rd.seed(config.cur_seed)

        # Run the algorithm and return the statistics, the final population and the best runs (hall of fame).
        from algorithms.experimental import experimental_rademacher_complexity_v2 as algorithm
        statistics, population, halloffame = algorithm.execute_algorithm()

        # Outputs the statistics to a csv file which can be found in the output folder of the project.
        output_to_file(output_path, "experimental-rademacher-complexity", i+1, data_name,
                       statistics, rademacher_columns)

        print("=== Experimental Rademacher Complexity Execution " + str(i+1) + " Completed ===")


if __name__ == "__main__":
    main()
