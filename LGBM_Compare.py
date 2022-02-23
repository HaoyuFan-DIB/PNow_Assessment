# To test how changing some parameters will affect the accuracy and time-cost
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from Utils import load_csv_data, DataPath, ImgPath, ModelPath

par2test_all = {'boosting_type': ['gbdt', 'rf'],
               'num_leaves': [31, 15, 63, 127, 255, 511],
               'learning_rate': [0.1, 0.2, 0.05, 0.02, 0.01],
               }

params_std = {'objective': 'regression',
              'max_iter': 10000,
              'early_stopping_round': 5,
              'verbosity': 0,
              'force_row_wise': True,
              'bagging_fraction': 0.8,
              'bagging_freq': 5,
              'boosting_type': 'gbdt',
              'num_leaves': 31,
              'learning_rate': 0.1,
              'linear_tree': False
              }


# loading data
df_train, df_test, X_columns, Y_column = load_csv_data(d_band=True, norm_T=True)
lgb_train = lgb.Dataset(df_train[X_columns], df_train[Y_column])
# standard model
start_time = time.time()
gbm_std = lgb.train(params_std,
                    lgb_train,
                    valid_sets=lgb.Dataset(df_test[X_columns], df_test[Y_column], reference=lgb_train))

duration = time.time() - start_time
train_score = gbm_std.best_score['valid_0']['l2']
y_pred = gbm_std.predict(df_test.drop(columns=Y_column), num_iteration=gbm_std.best_iteration)
test_score = mean_squared_error(df_test[Y_column], y_pred)

df_standard = pd.DataFrame(columns=["Time", "l2_train", "l2_test"])
df_standard = df_standard.append({"Time": duration,
                                  "l2_train": train_score ** 0.5,
                                  "l2_test": test_score ** 0.5},
                                 ignore_index=True)

for par2test in par2test_all.keys():
    print("\n"*2 + "Now Testing " + par2test + "\n"*2)
    params = params_std
    result_df = df_standard
    result_df["par"] = par2test
    result_df["value"] = params[par2test]

    for value in par2test_all[par2test][1:]:
        params[par2test] = value
        print(params)

        start_time = time.time()
        gbm_test = lgb.train(params,
                             lgb_train,
                             valid_sets=lgb.Dataset(df_test[X_columns], df_test[Y_column], reference=lgb_train))

        duration = time.time() - start_time
        train_score = gbm_test.best_score['valid_0']['l2']
        y_pred = gbm_test.predict(df_test.drop(columns=Y_column), num_iteration=gbm_test.best_iteration)
        test_score = mean_squared_error(df_test[Y_column], y_pred)

        result_slice = {"Time": duration,
                        "l2_train": train_score ** 0.5,
                        "l2_test": test_score ** 0.5,
                        "par": par2test,
                        "value": value}
        result_df = result_df.append(result_slice, ignore_index=True)

    #plt.plot(result_df["Time"].to_numpy(), result_df["l2_train"].to_numpy(), label="train")
    # No over fitting so hide training
    plt.plot(result_df["Time"].to_numpy(), result_df["l2_test"].to_numpy(), label="test")
    for par2plot in result_df[["Time", "l2_train", "l2_test", "value"]].values.tolist():
        x, y1, y2, value = par2plot[0], par2plot[1], par2plot[2], par2plot[3]
        plt.scatter([x, x], [y1, y2], label=str(value))

    plt.grid()
    plt.legend(loc='upper right')
    plt.title(par2test)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Error")

    plt.savefig(os.path.join(ImgPath, par2test + ".png"))
    plt.close()







