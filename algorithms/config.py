import pandas as pd

# Number of executions of the algorithm.
executions = 100

# Genetic Operators.
prob_crossover = 0.9
prob_mutation = 0.1
prob_elitism = 0.1

# Max height of genetic operators.
max_height = 10

# Population and Generations.
num_generations = 100
size_population = 1024

# Random terminal limits.
random_upper = 1
random_lower = -1

# Train-Test split size.
split_size = 0.3

# Path to reproducibility seeds.
seed = "../data/seed.csv"

# Loading the seed values from file.
seeds = pd.read_csv(seed, header=None)

# Path to regression data-sets.
bhouse_train = "../data/ccun/bhouse-training.csv"
bhouse_test = "../data/ccun/bhouse-testing.csv"

ccun_train = "../data/ccun/ccun-training.csv"
ccun_test = "../data/ccun/ccun-testing.csv"

cd_train = "../data/cd/cd-training.csv"
cd_test = "../data/cd/cd-testing.csv"

dlbcl_train = "../data/ld50/dlbcl-training.csv"
dlbcl_test = "../data/ld50/dlbcl-testing.csv"

ld50_train = "../data/ld50/ld50-training.csv"
ld50_test = "../data/ld50/ld50-testing.csv"

wine_train = "../data/ld50/wine-training.csv"
wine_test = "../data/ld50/wine-testing.csv"

training_data = pd.DataFrame()
testing_data = pd.DataFrame()

cur_seed = 0
