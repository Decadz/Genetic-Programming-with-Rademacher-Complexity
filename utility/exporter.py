
benchmark_classic_path = "../output/benchmark_classic/"

def output_to_file(path, filename, filenumber, datasetnumber, dataframe):
    # TODO - Need to implement so it saves as a csv file to the correct folder.
    # Want seed number and dataset number in file name.

    # Creating and prepare file for writing.
    f = open(path + filename + str(filenumber) + ".txt", "w+")
