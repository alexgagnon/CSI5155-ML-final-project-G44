# CSI5155 Final Project

The purpose of this repository is to use ML algorithms to attempt to classify VPN vs. non-VPN traffic. The datasets used come from the University of New Brunswick Cybersecurity department.

## FILES

- main.py - the main source code to run
- util.py - supplementary module providing additional functions
- results/ - directory that will be produced when you run main.py (must have write permissions)
- results/output.log - file containing the log output of the results
- results/{dataset}-{classifier}-{feature_selection}.txt - the results of each classifier

If the code is configured to report EDA, CV results, etc., these will show up in the 'results/' folder

## How to use

Running `main.py` will produce results. You can modify the source code to easily change the number/type of classifiers, which reports will be generated, feature selections and feature engineering methods, scoring metrics, etc. These are listed as constants at the top of the file.

## Requirements

Ideally you will have Anaconda installed, the required modules are:

- Orange
- sklearn
- skmultiflow
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- tabulate
