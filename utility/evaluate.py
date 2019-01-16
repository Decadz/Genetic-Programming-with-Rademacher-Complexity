import numpy as np
from math import sqrt

import algorithms.config as config

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


def evaluate_population(generation_num, generation_time, populations_fitnesses, populations_sizes, individual, func):

    statistics = [generation_num + 1, generation_time]

    fitness_min = np.percentile(populations_fitnesses, 0)
    fitness_lq = np.percentile(populations_fitnesses, 25)
    fitness_med = np.percentile(populations_fitnesses, 50)
    fitness_uq = np.percentile(populations_fitnesses, 75)
    fitness_max = np.percentile(populations_fitnesses, 100)
    fitness_std = np.std(populations_fitnesses)
    fitness_mean = np.mean(populations_fitnesses)

    # Adding all the statistics related to the populations fitness (error).
    statistics.extend([fitness_min, fitness_lq, fitness_med, fitness_uq, fitness_max, fitness_std, fitness_mean])

    size_min = np.percentile(populations_sizes, 0)
    size_lq = np.percentile(populations_sizes, 25)
    size_med = np.percentile(populations_sizes, 50)
    size_uq = np.percentile(populations_sizes, 75)
    size_max = np.percentile(populations_sizes, 100)
    size_std = np.std(populations_sizes)
    size_mean = np.mean(populations_sizes)

    # Adding all the statistics related to the populations size.
    statistics.extend([size_min, size_lq, size_med, size_uq, size_max, size_std, size_mean])

    pred = []  # List of predicted values.
    real = []  # List of ground truth values.

    for rows in range(config.testing_data.shape[0]):
        pred.append(func(*(config.testing_data.values[rows][0:config.testing_data.shape[1] - 1])))
        real.append(config.testing_data.values[rows][config.testing_data.shape[1] - 1])

    mse = mean_squared_error(real, pred)
    rmse = sqrt(mean_squared_error(real, pred))
    r2 = r2_score(real, pred)
    mae = median_absolute_error(real, pred)

    # Solving the relative squared error (RAE).
    squared_error = 0
    relative_error = 0

    for i in range(len(real)):
        squared_error += (pred[i] - real[i]) ** 2
        relative_error += (sum(real)/len(real) - real[i]) ** 2

    rse = squared_error/relative_error

    size = len(individual)

    statistics.extend([mse, rmse, r2, mae, rse, size, str(individual)])

    return statistics


def evaluate_population_rademacher(generation_num, generation_time, populations_fitnesses, population_errors,
                                   population_bounds, populations_sizes, individual, func):

    statistics = [generation_num + 1, generation_time]

    fitness_min = np.percentile(populations_fitnesses, 0)
    fitness_lq = np.percentile(populations_fitnesses, 25)
    fitness_med = np.percentile(populations_fitnesses, 50)
    fitness_uq = np.percentile(populations_fitnesses, 75)
    fitness_max = np.percentile(populations_fitnesses, 100)
    fitness_std = np.std(populations_fitnesses)
    fitness_mean = np.mean(populations_fitnesses)

    # Adding all the statistics related to the populations fitness (error + rademacher penalty).
    statistics.extend([fitness_min, fitness_lq, fitness_med, fitness_uq, fitness_max, fitness_std, fitness_mean])

    errors_min = np.percentile(population_errors, 0)
    errors_lq = np.percentile(population_errors, 25)
    errors_med = np.percentile(population_errors, 50)
    errors_uq = np.percentile(population_errors, 75)
    errors_max = np.percentile(population_errors, 100)
    errors_std = np.std(population_errors)
    errors_mean = np.mean(population_errors)

    # Adding all the statistics related to the populations errors.
    statistics.extend([errors_min, errors_lq, errors_med, errors_uq, errors_max, errors_std, errors_mean])

    bounds_min = np.percentile(population_bounds, 0)
    bounds_lq = np.percentile(population_bounds, 25)
    bounds_med = np.percentile(population_bounds, 50)
    bounds_uq = np.percentile(population_bounds, 75)
    bounds_max = np.percentile(population_bounds, 100)
    bounds_std = np.std(population_bounds)
    bounds_mean = np.mean(population_bounds)

    # Adding all the statistics related to the populations fitness (error).
    statistics.extend([bounds_min, bounds_lq, bounds_med, bounds_uq, bounds_max, bounds_std, bounds_mean])

    size_min = np.percentile(populations_sizes, 0)
    size_lq = np.percentile(populations_sizes, 25)
    size_med = np.percentile(populations_sizes, 50)
    size_uq = np.percentile(populations_sizes, 75)
    size_max = np.percentile(populations_sizes, 100)
    size_std = np.std(populations_sizes)
    size_mean = np.mean(populations_sizes)

    # Adding all the statistics related to the populations size.
    statistics.extend([size_min, size_lq, size_med, size_uq, size_max, size_std, size_mean])

    pred = []  # List of predicted values.
    real = []  # List of ground truth values.

    for rows in range(config.testing_data.shape[0]):
        pred.append(func(*(config.testing_data.values[rows][0:config.testing_data.shape[1] - 1])))
        real.append(config.testing_data.values[rows][config.testing_data.shape[1] - 1])

    mse = mean_squared_error(real, pred)
    rmse = sqrt(mean_squared_error(real, pred))
    r2 = r2_score(real, pred)
    mae = median_absolute_error(real, pred)

    # Solving the relative squared error (RAE).
    squared_error = 0
    relative_error = 0

    for i in range(len(real)):
        squared_error += (pred[i] - real[i]) ** 2
        relative_error += (sum(real) / len(real) - real[i]) ** 2

    rse = squared_error / relative_error

    size = len(individual)

    statistics.extend([mse, rmse, r2, mae, rse, size, str(individual)])

    return statistics
