import os
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Paths
RootPath = os.path.dirname(os.path.realpath(__file__))
DataPath = os.path.join(RootPath, "Data")
ImgPath = os.path.join(RootPath, 'IMG')
if not os.path.exists(ImgPath):
    os.mkdir(ImgPath)

ModelPath = os.path.join(RootPath, 'Model')
if not os.path.exists(ModelPath):
    os.mkdir(ModelPath)

# Ranges of stellar parameters
teff_range = [3000, 7000]
logg_range = [1, 5]
feh_range = [-2, 0.5]


# Function for to get input data
def load_csv_data(test_size=0.25, d_band=True, norm_T=True):
    """
    Function to load the training and testing dataset from DataPath
    :param d_band: bool, if true, use difference between brightness, otherwise load raw brightness
    :param norm_T: bool, if true, load normalized TEFF, otherwise use raw temperature
    :return: df_training, df_testing, X_columns, Y_column, as specified by inputs
    """
    if d_band:
        X_columns = ['r-g', 'i-g', 'i-r', 'b-g', 'b-r', 'b-i', 'v-g',
                     'v-r', 'v-i', 'v-b', 'j-g', 'j-r', 'j-i', 'j-b',
                     'j-v', 'h-g', 'h-r', 'h-i', 'h-b', 'h-v', 'h-j']
    else:
        X_columns = ["mag_" + item for item in "gribvjh"]

    if norm_T:
        Y_column = ["teff_norm"]
    else:
        Y_column = ["teff"]

    csv_filename = os.path.join(DataPath, "TrimmedData.csv")
    df = pd.read_csv(csv_filename, delimiter=",")
    df_train, df_test = train_test_split(df, test_size=test_size)

    return df_train[X_columns + Y_column], df_test[X_columns + Y_column], X_columns, Y_column