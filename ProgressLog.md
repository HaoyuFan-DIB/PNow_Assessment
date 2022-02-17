# Daily Progress Log for PNow Assessment Project
2020-02-15
First day of the project, review requirement document and break it down to bullet points to work on. Detailed progress including:
- Created Git repo and will continuously update;
- Data preprocessing: data trimming done, working on EDA and feature engineering, should be done by tomorrow;
- Model training (NN): retrieved old script using Keras. The NN should be re-designed since I'll be only focusing on one parameter (instead of three). So 64 node * 8 layers? Will train the model tomorrow night.
- Model training (lightGBM): read `lightGBM` document and basic example script, seems straight-forward. First model should be ready by the end of tomorrow;
- Distributed Computation: read `Dask` document and a little confusing. It's like parallel computation but how to cope with model training? Training a subset of trees on each worker and combine them into a forest?
- Deploy model: no idea by far;
- Cope with new data: some idea. Assuming we have a forest of trees, and the old tress would work the same on old data regardless of the new data. So instead of chopping down the entire forest, maybe I should just add a few new tress from the new data? Should I oversample the new data?

2020-02-16
Second day of the project, made some progress and reached first result.
- Data pre-processing: done, also a util function to help load data quickly;
- Model training (lightGBM): followed example script on website and it run through. Super quick though less accurate than NN as I remember. First result ready but lots of room for fine-tuning;
- Model training (NN): delayed, but should be straight forward and training should be less intense;
- Distributed computation: found an example script on using lgbm with `Dask`, will catch up for detail tomorrow;
