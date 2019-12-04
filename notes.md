PRESENTATION NOTES:

- wanted to use a diverse set of algorithms
- wanted to see if we could set up a pipe
- was easier to not use pipelines
- wanted to compute multiple stats on each cross validation, but also to keep things consistent between runs and between algorithms, so used Stratified folds
- did not do hyperparameter optimization
- had trouble with doing deep copies of dataframes, which would have sped up the script significantly (i.e. only normalize once). Appears you can't
- could not get hoeffding tree to work similarly to the sklearn ones
- could not get cn2 from Orange to work correctly
