import pandas as pd

# Number of executions of the algorithm.
executions = 1

# Genetic Operators.
prob_crossover = 0.9
prob_mutation = 0.1

# Max height of genetic operators.
max_height = 20

# Population and Generations.
num_generations = 300
size_population = 300

# Random terminal limits.
random_upper = 5
random_lower = 1

# Train-Test split size.
split_size = 0.3

# Path to reproducibility seeds.
seed = "../data/seed.csv"

# Loading the seed values from file
seeds = pd.read_csv(seed, header=None)

# Path to regression data-sets.
ccn_train = "../data/ccn/ccn-training.csv"
ccn_test = "../data/ccn/ccn-testing.csv"

ccun_train = "../data/ccun/ccun-training.csv"
ccun_test = "../data/ccun/ccun-testing.csv"

cd_train = "../data/cd/cd-training.csv"
cd_test = "../data/cd/cd-testing.csv"

dlbcl_train = "../data/dlbcl/dlbcl-training.csv"
dlbcl_test = "../data/dlbcl/dlbcl-testing.csv"

ld50_train = "../data/ld50/ld50-training.csv"
ld50_test = "../data/ld50/ld50-testing.csv"

experimental_train_1 = "../data/experimental/experimental-training-1.csv"
experimental_test_1 = "../data/experimental/experimental-testing-1.csv"

experimental_train_2 = "../data/experimental/experimental-training-2.csv"
experimental_test_2 = "../data/experimental/experimental-testing-2.csv"

training_data = pd.DataFrame()
testing_data = pd.DataFrame()

cur_seed = 0
