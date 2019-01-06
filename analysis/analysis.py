__author__ = "Christian Raymond"
__date__ = "04 January 2019"

"""
Script for analysing the output data of the various genetic programming 
implementations. Outputs data to the analysis folder.
"""

import pandas as pd
import matplotlib.pyplot as plt


num_samples = 50

classic_cd_path = "../output/benchmark_classic/benchmark-classic-cd-"
classic_ccn_path = "../output/benchmark_classic/benchmark-classic-ccn-"
classic_ccun_path = "../output/benchmark_classic/benchmark-classic-ccun-"
classic_ld50_path = "../output/benchmark_classic/benchmark-classic-ld50-"

rademacher_cd_path = "../output/experimental_rademacher_complexity/experimental-rademacher-complexity-cd-"
rademacher_ccn_path = "../output/experimental_rademacher_complexity/experimental-rademacher-complexity-ccn-"
rademacher_ccun_path = "../output/experimental_rademacher_complexity/experimental-rademacher-complexity-ccun-"
rademacher_ld50_path = "../output/experimental_rademacher_complexity/experimental-rademacher-complexity-ld50-"


def main():

    # Loading the Genetic Programming classic output data.
    classic_cd = load_data(classic_cd_path)
    classic_ccn = load_data(classic_ccn_path)
    classic_ccun = load_data(classic_ccun_path)
    classic_ld50 = load_data(classic_ld50_path)

    # Loading the Genetic Programming with Rademacher Complexity output data.
    rademacher_cd = load_data(rademacher_cd_path)
    rademacher_ccn = load_data(rademacher_ccn_path)
    rademacher_ccun = load_data(rademacher_ccun_path)
    rademacher_ld50 = load_data(rademacher_ld50_path)

    # Analysing the Genetic Programming classic output data.
    bc1 = analyse(classic_cd, "benchmark-classic", "cd")
    bc2 = analyse(classic_ccn, "benchmark-classic", "ccn")
    bc3 = analyse(classic_ccun, "benchmark-classic", "ccun")
    bc4 = analyse(classic_ld50, "benchmark-classic", "ld50")

    # Analysing the Genetic Programming with Rademacher Complexity output data.
    er1 = analyse(rademacher_cd, "experimental-rademacher", "cd")
    er2 = analyse(rademacher_ccn, "experimental-rademacher", "ccn")
    er3 = analyse(rademacher_ccun, "experimental-rademacher", "ccun")
    er4 = analyse(rademacher_ld50, "experimental-rademacher", "ld50")

    visualise(bc1, "benchmark-classic", er1, "experimental-rademacher", "CD")
    visualise(bc2, "benchmark-classic", er2, "experimental-rademacher", "CCN")
    visualise(bc3, "benchmark-classic", er3, "experimental-rademacher", "CCUN")
    visualise(bc4, "benchmark-classic", er4, "experimental-rademacher", "LD50")


def load_data(output_path):

    """
    Loads all the data for a specific algorithm, loading all n number of runs.

    :param output_path:
    :return:
    """

    paths = list()

    # Creating a list of paths to load the output data.
    for index in range(num_samples):
        paths.append(output_path + str(index + 1) + ".csv")

    data = list()

    # Loading the data and putting it in a list.
    for index in range(len(paths)):
        data.append(pd.read_csv(paths[index]))

    return data


def analyse(data, algorithm_name, data_name):

    """
    Merges the data (taking the averages across all runs) and than extracts the key information.
    The pandas dataframe in exported to a csv file which appears in the analysis folder.

    :param data:
    :param algorithm_name:
    :param data_name:
    :return:
    """

    gen_num = list()
    gen_time = list()
    training_fitness = list()
    testing_fitness = list()
    size_med = list()

    # Extracting the important columns.
    for index in range(num_samples):

        gen_num.append(data[index][["generation_num"]])
        gen_time.append(data[index][["generation_time"]])
        training_fitness.append(data[index][["training_fitness_min"]])
        testing_fitness.append(data[index][["test_mse"]])
        size_med.append(data[index][["size_med"]])

    result = list()

    gen_num = sum(gen_num)/num_samples
    gen_time = sum(gen_time)/num_samples
    training_fitness = sum(training_fitness)/num_samples
    testing_fitness = sum(testing_fitness)/num_samples
    size_med = sum(size_med)/num_samples

    for i in range(len(gen_num)):

        result.append([gen_num.iloc[i].values[0], gen_time.iloc[i].values[0], training_fitness.iloc[i].values[0],
                testing_fitness.iloc[i].values[0], size_med.iloc[i].values[0]])


    df = pd.DataFrame(result)
    df.columns = ["Generation Number", "Generation Time", "Training Error", "Testing Error", "Average Size"]
    df.to_csv(algorithm_name + "-" + data_name + ".csv", index=False)

    return df


def visualise(df1, df1_name, df2, df2_name, dataset_name):

    """
    Creates multiple plots comparing two algorithms across n number of generations.

    :param df1: The first dataframe (based on algorithm 1).
    :param df1_name: The algorithms name which produced the data.
    :param df2: The second dataframe (based on algorithm 2).
    :param df2_name: The algorithms name which produced the data.
    :param dataset_name: The dataset this data was produced on.
    :return: Plot comparing all of the errors across generations.
    """

    visualise_error(df1, df1_name, df2, df2_name, dataset_name)
    visualise_size(df1, df1_name, df2, df2_name, dataset_name)

"""
===============================================================================================================

    - Visualisations 

===============================================================================================================
"""


def visualise_error(df1, df1_name, df2, df2_name, dataset_name):

    """
    Creates a plot comparing two algorithms error across n number of generations.

    :param df1: The first dataframe (based on algorithm 1).
    :param df1_name: The algorithms name which produced the data.
    :param df2: The second dataframe (based on algorithm 2).
    :param df2_name: The algorithms name which produced the data.
    :param dataset_name: The dataset this data was produced on.
    """

    # Giving the plot a title.
    plt.title("Classical GP vs Rademacher GP (" + dataset_name + ")")

    # Labelling the plots axis's.
    plt.xlabel("Generation")
    plt.ylabel("Error")

    # Calculating what the biggest error is so that the plot regions can be defined.
    error1 = df1[df1["Training Error"] == df1["Training Error"].max()]
    error2 = df2[df2["Training Error"] == df2["Training Error"].max()]
    error3 = df1[df1["Testing Error"] == df1["Testing Error"].max()]
    error4 = df2[df2["Testing Error"] == df2["Testing Error"].max()]

    # Setting the range of the visible axis's.
    plt.axis([0, 250, 0, max(error1["Training Error"].values[0], error2["Training Error"].values[0],
                             error3["Testing Error"].values[0], error4["Testing Error"].values[0])])

    # Plotting both the training and testing error for dataframe 1.
    plt.plot(df1[["Generation Number"]], df1["Training Error"], label=df1_name + "-training")
    plt.plot(df1[["Generation Number"]], df1["Testing Error"], label=df1_name + "-testing")

    # Plotting both the training and testing error for dataframe 2.
    plt.plot(df2[["Generation Number"]], df2["Training Error"], label=df2_name + "-training")
    plt.plot(df2[["Generation Number"]], df2["Testing Error"], label=df2_name + "-testing")

    # Where the legend (key) is located on the diagram.
    plt.legend(loc="upper right")

    # Saving the diagram to the analysis folder (CWD).
    plt.savefig("classic-vs-rademacher-error-" + dataset_name + ".png")
    plt.close()


def visualise_size(df1, df1_name, df2, df2_name, dataset_name):

    """
    Creates a plot comparing two algorithms program size across n number of generations.

    :param df1: The first dataframe (based on algorithm 1).
    :param df1_name: The algorithms name which produced the data.
    :param df2: The second dataframe (based on algorithm 2).
    :param df2_name: The algorithms name which produced the data.
    :param dataset_name: The dataset this data was produced on.
    """

    # Giving the plot a title.
    plt.title("Classical GP vs Rademacher GP (" + dataset_name + ")")

    # Labelling the plots axis's.
    plt.xlabel("Generation")
    plt.ylabel("Size")

    # Calculating what the biggest size is so that the plot regions can be defined.
    size1 = df1[df1["Average Size"] == df1["Average Size"].max()]
    size2 = df2[df2["Average Size"] == df2["Average Size"].max()]

    # Setting the range of the visible axis's.
    plt.axis([0, 250, 0, max(size1["Average Size"].values[0], size2["Average Size"].values[0])])

    # Plotting the program size for dataframe 1.
    plt.plot(df1[["Generation Number"]], df1["Average Size"], label=df1_name)

    # Plotting the program size for dataframe 2.
    plt.plot(df2[["Generation Number"]], df2["Average Size"], label=df2_name)

    # Where the legend (key) is located on the diagram.
    plt.legend(loc="upper right")

    # Saving the diagram to the analysis folder (CWD).
    plt.savefig("classic-vs-rademacher-size-" + dataset_name + ".png")
    plt.close()


if __name__ == "__main__":
    main()
