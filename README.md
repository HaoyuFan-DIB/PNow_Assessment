## PNow_Assessment
The coding assessment for PNow internship. I will be using the stellar parameter data set and predict the temperature of stellar objects from their brightness and colors. The key aspects of this projects include:
- Feature engineering and EDA report while preparing the data;
- Train the model using `lightGBM` framework;
- Turn the pipeline into a distributed version using `Dask` or `Ray`;
- Share the results with colleagues at PNow.

### About the Data
The data is the star catalog from LAMOST data release 8. I will be using the `mag1` - `mag7` columns to predict the `teff` column. Some key aspects of the data set include:
- The raw data has ~1.3 M rows, and the filtered data has ~0.34 M rows. That leaves ~250K or training and 85K for testing (25% of testing data).
- The typical/median error of `teff` is 90K, or 0.023 after normalization.
- The filtering effort include: `SNR` of all band being positive and max(`SNR`) > 20; target being a star; `teff` between 3k - 7k, `logg` between 1.0 - 5.0, `feh` between -2.0 - 0.5.
- The `magnitude type` (i.e. `mag1` - `mag7`) being *g-r-i-b-v-j-h*.

### ANN Models
Before jumping to LGBM, I worked with some ANN models for benchmark purpose. This is similar to my previous project expect I am just predicting `teff` this time.
- The time cost of training ANN models greatly depends on the total epochs. Most of the improvements happen within the first few epochs.
- I find 500 - 1000 epochs to be quite sufficient, and each epoch takes about 5 - 7 sec, so the training takes 0.5 - 2.0 hr.
- Depending on the complexity of the ANN model, RMSE can be ~0.026, which is close to the error of data, and better than LGBM (see below).

### LGBM
Using LGBM following the documentation is not hard, but there are many parameters to tune with. Some key points I found include:
- LGBM is faster than ANN, but the result is less accurate, where the RMSE is ~ 0.035.
- The fast-training process usually takes less than 1s, compared to ~30s in ANN.
- There are many parameters to tune, some deals with overfitting while the others balance efficiency and effectiveness. 
- Over fitting is not a problem in my data.
- Tried to play with `num_of_leaf`, `learn_rate`, `boosting_type`, and the default value is the best?
- Feature engineering is successful. RMSE is smaller when use `d_mag`, compared to use the raw `mags`.
- Basic LGBM got RMSE = 0.0339 from 1000 iteration, using 6.75s; Distributed LGBM got RMSE = 0.0392 using 8.71s??
- And trouble using eval_set in DaskLGBMRegressor...
