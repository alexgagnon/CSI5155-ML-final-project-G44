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

- goal was to see if we could match or increase the findings of several papers by using different algorithms
- important because VPN traffic is considered secure. Being able to identify traffic could be used to target individuals in places where privacy and human rights run counter to the objectives of the state. While this makes it seem to be counter intuitive, as we are trying to find good performing models, it is useful to the end user to know that their traffic can be detected.
- studying literature, we found several other papers attempts to identify traffic (both in the sense of detecting VPN vs. non-VPN traffic, and detecting the type of traffic, such as skype, browsing, streaming, etc.) - a large effort was placed into making sure the processing for each algorithm was identical to be able to determine statistical significance. In a real world use case, this is not optimal as we simply want the best performing algorithm, regardless of how the model is created.
- addressed by using multiple datasets and various algorithms to identify well performing ones: - We first collected data from UNB, who collected the initial dataset by capturing streams of traffic using either a VPN connection or directly on the Internet, and with various applications (i.e. browsing, skype, etc.) - 4 datasets of various timeouts (15s, 30s, 60s, 120s). In our effort to keep the datasets similar, we removed the 60s dataset as it contained different features - prilimary EDA of the remaining 3 datasets showed: - the classes were balanced in each - there was a large number of missing values, denoted by a -1, which we removed - unlike our primary literature source, we did not remove duplicates, as we felt these we unlikely to be noise, but real values, and as such were valuable to the model training -

- implementation issues - lack of knowledge of python meant we struggled a lot with setting up the code base. For example, we weren't able to get Hoeffding Tree and CN2 algorithm to work, as the differing package structure and results weren't condusive to the rest of the sklearn classifiers
