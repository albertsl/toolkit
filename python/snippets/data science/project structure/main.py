# Main file of the project. This file is pretended to run the code.
# Author: Albert Sanchez
# June 2018

from data_importer import Data_Importer
from data_processor import Data_Processor
from model import Model

from os import path

def main():
    train_file = path.abspath("train.csv")
    test_file = path.abspath("test.csv")
    di = Data_Importer()
    df = di.get_data(train_file)

    dp = Data_Processor()
    dfp = dp.process_data(df)

    m = Model()
    m.model_data(dfp)

if __name__ == "__main__":
    main()
