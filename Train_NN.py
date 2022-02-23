import pandas as pd
import numpy as np
import time
import os
from Utils import load_csv_data, DataPath, ImgPath, ModelPath

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from tensorflow.keras.optimizers import Adam

# util functions
def get_model_dense(n_filter, n_layer_group, activation, final_activation):
    input_layer = Input([21])

    next_layer = Dense(n_filter, kernel_initializer='he_uniform')(input_layer)


    for i in range(n_layer_group):
        next_layer = get_model_group(next_layer, n_filter, activation)

    final_layer = BatchNormalization()(next_layer)
    final_layer = Activation(activation)(final_layer)
    final_layer = Dense(1, kernel_initializer='he_uniform', activation=final_activation, name="teff")(final_layer)

    return Model(input_layer, final_layer)


def get_model_group(input_layer, n_filter, activation):
    for i in range(2):
        out_layer = BatchNormalization()(input_layer)
        out_layer = Activation(activation=activation)(out_layer)
        out_layer = Dense(n_filter, kernel_initializer='he_uniform')(out_layer)

    #out_layer = add([out_layer, input_layer])

    return out_layer


class MyMetrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        # self.validation_data = None
        self.model = None
        self.best_val_loss = 9999
        self.min_teff_error_rms = 9999
        self.min_teff_error_mae = 9999
        self.start_time = time.time()
        self.loss_log = pd.DataFrame(columns=["epoch", "time", "RMS", "MAE"])


    def on_epoch_end(self, epoch, logs=None):
        teff_pred = self.model.predict(self.validation_data[0])
        teff_real = self.validation_data[1]

        teff_error_rms = np.sqrt(np.sum(np.square(teff_pred - teff_real)) / len(teff_real))
        teff_error_mae = np.mean(np.abs(teff_pred - teff_real))

        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            self.min_teff_error_rms = teff_error_rms
            self.min_teff_error_mae = teff_error_mae

            print("\n" + "="*50)
            print("New Best!")
            print("RMS = %.4f" % teff_error_rms)
            print("MAE = %.4f" % teff_error_mae)
            print("=" * 50 + "\n")

            with open(os.path.join(ModelPath, "LossVSTime.csv"), "w") as f:
                result_slice = {"epoch": epoch,
                                "time": time.time() - self.start_time,
                                "RMS": teff_error_rms,
                                "MAE": teff_error_mae}
                self.loss_log = self.loss_log.append(result_slice, ignore_index=True)
                f.write(self.loss_log.to_csv(index=False))


def my_lr(epoch, lr):
    if epoch <= 5:
        lr = 5e-3
    elif epoch <= 50:
        lr = 1e-3
    elif epoch <= 200:
        lr = 1e-4
    else:
        lr = 1e-5
    return lr



# basic parameters
if __name__ == '__main__':
    batch_size = 1024
    epochs = 1000
    n_filter = 128
    n_layer_Group = 4
    activation = 'relu'
    final_activation = 'linear'

    df_train, df_test, X_columns, Y_column = load_csv_data(d_band=True, norm_T=True)
    x_train = df_train[X_columns].to_numpy()
    y_train = df_train[Y_column].to_numpy()
    x_test = df_train[X_columns].to_numpy()
    y_test = df_train[Y_column].to_numpy()

    model = get_model_dense(n_filter, n_layer_Group, activation, final_activation)
    # model = load_model('./best_model_lamost.h5')
    model.summary()

    model.compile(Adam(1e-3, amsgrad=True), ['mse'])
    model.fit(x_train, y_train,
              validation_data=[x_test, y_test],
              batch_size=batch_size, epochs=epochs, verbose=1,
              callbacks=[MyMetrics([x_test, y_test]),
                         LearningRateScheduler(my_lr, verbose=1)],
              initial_epoch=0)

    os.rename(os.path.join(ModelPath, "LossVSTime.csv"),
              os.path.join(ModelPath, "Loss_%iFilter_%iLayers.csv" % (n_filter, n_layer_Group)))



