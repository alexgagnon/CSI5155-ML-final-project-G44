# PRESENTATION NOTES

- wanted to use a diverse set of algorithms
- wanted to see if we could set up a the script so that makes it easy to compute and compare many algorithms and feature sets
- was easier to not use pipelines, due to not knowing sklearn sufficiently
- wanted to compute multiple stats on each cross validation, but also to keep things consistent between runs and between algorithms, so used Stratified folds

- differences between the 15s, 30s, and 120s datasets and the 60s dataset made us drop the 60s
- preliminary EDA lead to lots of domain information we used to do manual feature selection

* did not do hyperparameter optimization
* had trouble with doing deep copies of dataframes, which would have sped up the script significantly (i.e. only normalize once). Appears you can't
* could not get hoeffding tree to work similarly to the sklearn ones
* could not get cn2 from Orange to work correctly, so used Dummy rule classifier, which is not good

* deeper knowledge of Python and sklearn would have helped a lot
