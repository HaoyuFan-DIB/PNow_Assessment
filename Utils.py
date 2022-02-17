import os
import pandas as pd
import numpy as np
from Path import DataPath, ImgPath, ModelPath

def load_csv_data(d_band=True, norm_T=True):
    """
    Function to load the training and testing dataset from DataPath
    :param d_band: bool, if true, use difference between brightness, otherwise load raw brightness
    :param norm_T: bool, if true, load normalized TEFF, otherwise use raw temperature
    :return: df_training, df_testing, X_columns, Y_column, as specified by inputs
    """
    if d_band:
        X_columns = ['V-U', 'G-U', 'G-V', 'R-U', 'R-V', 'R-G', 'I-U', 'I-V', 'I-G', 'I-R']
    else:
        X_columns = [item + "MAG" for item in ['U', 'V', 'G', 'R', 'I']]

    if norm_T:
        Y_column = ["TEFF_Norm"]
    else:
        Y_column = ["TEFF"]

    df_train = pd.read_csv(os.path.join(DataPath, "TrainingData.csv"))
    df_test = pd.read_csv(os.path.join(DataPath, "TestingData.csv"))

    return df_train[X_columns + Y_column], df_test[X_columns + Y_column], X_columns, Y_column



RootPath = os.path.dirname(os.path.realpath(__file__))

DataPath = os.path.join(RootPath, "Data")

ImgPath = os.path.join(RootPath, 'IMG')
if not os.path.exists(ImgPath):
    os.mkdir(ImgPath)

ModelPath = os.path.join(RootPath, 'Model')
if not os.path.exists(ModelPath):
    os.mkdir(ModelPath)
