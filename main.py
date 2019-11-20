import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.trees import HoeffdingTree
from util import print_df_meta

PRINT_META = False
SHOW_PLOTS = False
SHOW_PLOTS_AS_SUBPLOT = False

FILE_PATH_DIR = './datasets/Scenario A1/'
FILES = {
    '15s': 'TimeBasedFeatures-Dataset-15s-VPN.arff',
    '30s': 'TimeBasedFeatures-Dataset-30s-VPN.arff',
    #  '60s': 'TimeBasedFeatures-Dataset-60s-VPN.arff', # dropped due to inconsistent features compared to the other datasets
    '120s': 'TimeBasedFeatures-Dataset-120s-VPN.arff'
}
CLASSIFIERS = {
    'decision_tree': DecisionTreeClassifier(),
    # 'hoeffding_tree': HoeffdingTree(),
    'kNN': KNeighborsClassifier()
}

for dataset_label, filename in FILES.items():
    dataset, meta = loadarff(FILE_PATH_DIR + filename)
    data = pd.DataFrame(dataset)

    # need to handle that the dataset uses -1 as 'null'
    # data = data.replace(-1, 0)

    X = data.drop('class1', axis=1)
    y = data['class1'].astype('str')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.34, random_state=42)

    if (PRINT_META):
        if (SHOW_PLOTS):
            # NOTE: plt.show(block=False) is async, need at least one plt.show()
            # to pause execution so you can read the graph

            # distribution of values in each feature
            if (SHOW_PLOTS_AS_SUBPLOT):
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

    for classifier_name, classifier in CLASSIFIERS.items():
        steps = [
            ('normalize', Normalizer()),
            ('fit', classifier)
        ]
        pipeline = Pipeline(steps)

        pipeline.fit(X_train, y_train)
        # result = pipeline.predict(X_test)
        print(dataset_label + " + " + classifier_name +
              ": " + str(pipeline.score(X_test, y_test)))
