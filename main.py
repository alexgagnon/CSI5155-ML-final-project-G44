import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from util import print_df_meta

PRINT_META = True
SHOW_AS_SUBPLOT = True

file_path = './datasets/Scenario A1/TimeBasedFeatures-Dataset-15s-VPN.arff'

dataset, meta = loadarff(file_path)
data = pd.DataFrame(dataset)

# need to handle that the dataset uses -1 as 'null'
data.replace(-1, np.nan)
print_df_meta(data)
X = data.drop('class1', axis=1)
y = data['class1'].astype('str')

if (PRINT_META):
    # NOTE: plt.show(block=False) is async, need at least one plt.show()
    # to pause execution so you can read the graph
    # distribution of values in each feature
    if (SHOW_AS_SUBPLOT):
        fig, axes = plt.subplots(ncols=4, nrows=6, figsize=(15, 5))
        a = [i for i in axes for i in i]
        for i, ax in enumerate(a):
            sns.distplot(X[X.columns[i - 1]], ax=ax)
        plt.tight_layout()
        plt.show(block=False)
    else:
        for i in X.columns:
            plt.figure(figsize=(15, 5))
            sns.distplot(X[i])
            plt.show(block=False)

    # totals for classes
    cmap = sns.color_palette("Set2")
    sns.countplot(x='class1', data=data, palette=cmap)
    plt.xticks(rotation=45)
    plt.show(block=False)

    # correlation of features
    corr = data[data.columns[1:]].corr()
    plt.figure(figsize=(15, 8))
    sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 20))
    plt.show()

    print_df_meta(data)

steps = [('normalize', Normalizer()), ('fit', DecisionTreeClassifier())]
pipeline = Pipeline(steps)

pipeline.fit(X, y)
result = pipeline.predict(X)
print(pipeline.score(X, y))
