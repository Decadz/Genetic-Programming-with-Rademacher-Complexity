import pandas as pd


columns = ["generation_num", "generation_time", "fitness_min", "fitness_lq", "fitness_med", "fitness_uq", "fitness_max",
                  "fitness_std", "fitness_mean", "size_min", "size_lq", "size_med", "size_uq", "size_max", "size_std",
                  "size_mean", "mse", "rmse", "r2", "mae", "best_size", "best_solution"]


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
