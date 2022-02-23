# To test how changing some parameters will affect the accuracy and time-cost
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from Utils import load_csv_data, DataPath, ImgPath, ModelPath

# loading data
df_train, df_test, X_columns, Y_column = load_csv_data(d_band=True, norm_T=True)
lgb_train = lgb.Dataset(df_train[X_columns], df_train[Y_column])
lgb_test = lgb.Dataset(df_test[X_columns], df_test[Y_column], reference=lgb_train)

# Define Parameter
params = {
    'boosting_type': '',
    'objective': 'regression',
    #'metric': {'l2', 'l1'},
    'num_leaves': 63,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train model
start_time = time.time()
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_test
                )
duration = time.time() - start_time


y_pred = gbm.predict(df_test.drop(columns=Y_column), num_iteration=gbm.best_iteration)
rmse_test = mean_squared_error(df_test[Y_column].to_numpy(), y_pred) ** 0.5
print("RMSE = %.4f, Duration = %.2f" % (rmse_test, duration))