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
    # set parameters
    params = {
        'tree_learner': 'feature',
        'num_iterations': 10000,
        'early_stopping_round': 5,
        'num_leaves': 63,
        'learning_rate': 0.1,
        'verbose': 0
    }

    # get data
    print("loading data")
    df_train, df_test, X_columns, Y_column = load_csv_data(d_band=True, norm_T=True)
    x_train = df_train[X_columns].to_numpy()
    y_train = df_train[Y_column].to_numpy()
    x_test = df_train[X_columns].to_numpy()
    y_test = df_train[Y_column].to_numpy()

    dx = da.from_array(x_train, chunks=(22000, 21))
    dy = da.from_array(y_train, chunks=(22000, 1))

    print("initializing Dask cluster")
    cluster = LocalCluster(n_workers=2)
    clint = Client(cluster)


    dask_model = lgb.DaskLGBMRegressor(
                                       #boosting_type='rf',
                                       num_leaves=63,
                                       learning_rate=0.1,
                                       n_estimators=12,
                                       #feature_fraction=0.9,
                                       #bagging_fraction=0.8,
                                       #bagging_freq=5,
                                       tree_learner='data',
                                       num_iterations=1000,
                                       #early_stopping_round=5,
                                       )

    start_time = time.time()
    dask_model.fit(dx, dy,
                   #eval_metric='l2',
                   #eval_set=eval_set
                   )
    duration = time.time() - start_time

    y_pred = dask_model.predict(da.from_array(x_test))
    # eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print("RMSE = %.4f, Duration = %.2f" % (rmse_test, duration))





