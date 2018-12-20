import pandas as pd


columns = ["generation_num", "generation_time", "training_fitness_min", "training_fitness_lq", "training_fitness_med",
           "training_fitness_uq", "training_fitness_max", "training_fitness_std", "training_fitness_mean", "size_min",
           "size_lq", "size_med", "size_uq", "size_max", "size_std", "size_mean", "test_mse", "test_rmse", "test_r2",
           "test_mae", "best_candidate_size", "best_candidate_solution"]


def output_to_file(path, algorithm_name, file_number, data_name, statistics):

    """
    The statistics about a GP run is converted from a pandas data frame into to a csv
    file, and then saved in the desired location.

    :param path: Path to the folder where the file will be saved
    :param algorithm_name: Which algorithm is being tested
    :param file_number: The number of the experiment run
    :param data_name: The data set which is currently being used
    :param statistics: Pandas DataFrame containing the statistics
    """

    df = pd.DataFrame(statistics)
    df.columns = columns
    df.to_csv(path + algorithm_name + "-" + str(data_name) + "-" + str(file_number) + ".csv", index=False)
