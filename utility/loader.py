import pandas as pd


def load_data(path):

    """
    Loads the data into a Pandas Data Frame.
    :return: Pandas Data Frame
    """

    return pd.read_csv(path)