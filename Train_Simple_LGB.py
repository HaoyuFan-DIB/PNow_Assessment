import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from Utils import load_csv_data, DataPath, ImgPath, ModelPath


# loading data
df_train, df_test, X_columns, Y_column = load_csv_data(d_band=True, norm_T=True)

lgb_train = lgb.Dataset(df_train[X_columns], df_train[Y_column])
lgb_eval = lgb.Dataset(df_test[X_columns], df_test[Y_column], reference=lgb_train)

params = {
    'boosting_type': 'rf',
    'objective': 'regression',
    #'metric': {'l2', 'l1'},
    'num_leaves': 128,
    'learning_rate': 0.001,
    #'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'verbose': 1,
    'force_row_wise': True,
    'linear_tree': True
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])

print('Saving model...')
# save model to file
gbm.save_model(os.path.join(ModelPath, 'model.txt'))

print('Starting predicting...')
# predict
y_pred = gbm.predict(df_test.drop(columns=Y_column), num_iteration=gbm.best_iteration)
# eval
mse_test = mean_squared_error(df_test[Y_column], y_pred)
print(f'The MSE of prediction is: {rmse_test}')


# boosting: gbdt, rf
# num_iteration: default = 100. 20, 50, 100, 200?
# learning rate: default = 0.1, so 0.01, 0.05, 0.1, 0.2
# num_leaves: default=31, so 15, 63, 127? The range is 4000K with 100K accuracies there should be ~40 leaves?
# max_depth, no idea....
# min_data_in_leaf: default=20, seemed ok for the data set
# bagging_fraction/freq: avoid over-fitting, can turned off
# linear tree: linear model in each leaf rather than constant



# distributed:
# tree_learner: data for data_parallel?


