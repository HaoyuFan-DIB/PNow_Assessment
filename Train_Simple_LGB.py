import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from Utils import load_csv_data
from Path import DataPath, ImgPath, ModelPath


# loading data
df_train, df_test, X_columns, Y_column = load_csv_data(d_band=False, norm_T=False)
print(df_train)
print(df_test)
print(X_columns)
print(Y_column)

lgb_train = lgb.Dataset(df_train[X_columns], df_train[Y_column])
lgb_eval = lgb.Dataset(df_test[X_columns], df_test[Y_column], reference=lgb_train)

params = {
    'boosting_type': 'rf',
    'objective': 'regression',
    #'metric': {'l2', 'l1'},
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])

print('Saving model...')
# save model to file
gbm.save_model(os.path.join(ModelPath, 'model.txt'))

print('Starting predicting...')
# predict
y_pred = gbm.predict(df_test.drop(columns=Y_column), num_iteration=gbm.best_iteration)
# eval
rmse_test = mean_squared_error(df_test[Y_column], y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')

