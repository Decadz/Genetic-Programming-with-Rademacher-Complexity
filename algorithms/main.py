from deap import base
import pandas as pd
from utility.loader import load_data
from algorithms import benchmark_pareto_parsimony

from sklearn.model_selection import train_test_split


# Genetic Operators.
prob_crossover = 0.9
prob_mutation = 0.1

# Population and Generations.
num_generations = 30
size_population = 30

# Random terminal limits.
random_upper = 5
random_lower = 1

# Path to regression data-sets.
data_1 = "../data/regression-1.csv"
data_2 = "../data/regression-2.csv"
data_3 = "../data/regression-3.csv"
data_4 = "../data/regression-4.csv"
data_5 = "../data/regression-5.csv"


# All results should be sent to this output folder.
output_path = "../output/benchmark_pareto_parsimony/" #TODO - Fix Paths


data = load_data(data_1)
#seed = load_data(seed)


def main():
    train_data, test_data = train_test_split(data, test_size=0.3)

    # pop, stats, hof = benchmark_pareto_parsimony.main(train_data, test_data, 2018, base.Toolbox(), prob_crossover,
    #                            prob_mutation, num_generations, size_population, random_upper, random_lower)

    statistics, population, halloffame = benchmark_pareto_parsimony.main(train_data, test_data, 2018, base.Toolbox(),
                        prob_crossover, prob_mutation, num_generations, size_population, random_upper, random_lower)

    df = pd.DataFrame(statistics)
    df.to_csv(output_path + "temp.csv")

    # TODO - Get Execution time working, might need to add it to the generation eval so it can go to csv.
    # TODO - Load seeds and get them parsing correctly
    # TODO - Make exported csv have header names
    # TODO - Download regression data sets and put it into project path.
    # TODO - Implement README's in all the desired locations.
    # TODO - Get DEAP working at full speed, currently going back a py version.


def run_benchmark_classic(executions = 100):
    print("run")  # TODO - Implement


def run_benchmark_pareto_parsimony(executions = 100):
    print("run")  # TODO - Implement


if __name__ == "__main__":
    main()
