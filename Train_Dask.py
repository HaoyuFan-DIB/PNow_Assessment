import pandas as pd
import numpy as np
import time
import os
from Utils import load_csv_data, DataPath, ImgPath, ModelPath
from sklearn.metrics import mean_squared_error

import dask.array as da
from distributed import Client, LocalCluster
import lightgbm as lgb

if __name__ == "__main__":
    print("loading data")
    df_train, df_test, X_columns, Y_column = load_csv_data(d_band=True, norm_T=True)
    x_train = df_train[X_columns].to_numpy()
    y_train = df_train[Y_column].to_numpy()
    x_test = df_train[X_columns].to_numpy()
    y_test = df_train[Y_column].to_numpy()

    print(x_train.shape)
    print(y_train.shape)

    print("initializing Dask cluster")
    cluster = LocalCluster(n_workers=2)
    clint = Client(cluster)

    dx = da.from_array(x_train, chunks=(25000, 21))
    dy = da.from_array(y_train, chunks=(25000, 1))

    dask_model = lgb.DaskLGBMRegressor(n_estimators=10)
    dask_model.fit(dx, dy)

    y_pred = dask_model.predict(da.from_array(x_test))
    # eval
    rmse_test = mean_squared_error(y_test, y_pred)
    print(f'The RMSE of prediction is: {rmse_test}')





