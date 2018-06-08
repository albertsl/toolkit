# Class that encapsulates the gathering of the data.
# Author: Albert Sanchez
# May 2018

import pandas as pd

class Data_Importer:
    def __init__(self):
        pass

    def get_data(self, filename):
        """
        :param filename: string with the filename
        :return: pandas dataframe with the data
        """
        return pd.read_csv(filename)
