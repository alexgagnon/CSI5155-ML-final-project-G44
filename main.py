import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.trees import HoeffdingTree
from util import print_df_meta

PRINT_META = False
SHOW_NULLS = False
SHOW_SUMMARIES = True
SHOW_PLOTS = False
SHOW_PLOTS_AS_SUBPLOT = False
REMOVE_NULLS = True
REPLACE_NULL_VALUE = np.nan
SEED = 42
TARGET_CLASS = 'class1'
FILE_PATH_DIR = './datasets/Scenario A1/'
FILES = {
    '15s': 'TimeBasedFeatures-Dataset-15s-VPN.arff',
    '30s': 'TimeBasedFeatures-Dataset-30s-VPN.arff',
    # dropped due to inconsistent features compared to the other datasets
    #  '60s': 'TimeBasedFeatures-Dataset-60s-VPN.arff',
    '120s': 'TimeBasedFeatures-Dataset-120s-VPN.arff'
}
CLASSIFIERS = {
    'decision_tree': {
        'steps': [
            ('normalize', Normalizer()),
            ('fit', DecisionTreeClassifier(random_state=SEED))
        ],
        'parameters': {}
    },
    'hoeffding_tree': {
        'steps': [
            ('normalize', Normalizer()),
            ('fit', HoeffdingTree())
        ],
        'parameters': {}
    },
    'kNN': {
        'steps': [
            ('normalize', Normalizer()),
            ('fit', KNeighborsClassifier())
        ],
        'parameters': {}
    },
    'SGD (SVM)': {
        'steps': [
            ('normalize', Normalizer()),
            ('fit', SGDClassifier(random_state=SEED))
        ]
    },
    'k-means': {
        'steps': [
            ('normalize', Normalizer()),
            ('fit', KMeans(n_clusters=2, random_state=SEED))
        ]
    }
}
EXCLUDE_CLASSIFIERS = ['hoeffding_tree']
encoder = LabelEncoder()

for dataset_label, filename in FILES.items():
    dataset, meta = loadarff(FILE_PATH_DIR + filename)
    data = pd.DataFrame(dataset)

    # need to handle that the dataset uses -1 as 'null'
    if (REPLACE_NULL_VALUE != None):
        data = data.replace(-1, REPLACE_NULL_VALUE)
    nulls = data.isna().sum()
    if (REMOVE_NULLS):
        data = data.dropna(axis=0)
        data = data.reset_index(drop=True)
    if (PRINT_META and SHOW_NULLS):
        print('Null values')
        print('Before')
        print(nulls)
        print()
        print('After')
        print(data.isna().sum())

    # we can label encode as only the class is text, and it's binary
    # encoder.fit(data[TARGET_CLASS])
    # data[TARGET_CLASS] = encoder.transform(data[TARGET_CLASS])

    # extract the dataframe to data and target values
    X = data.drop(TARGET_CLASS, axis=1)
    y = data[TARGET_CLASS].astype('str')

    #  reporting...
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

        print_df_meta(data, dataset_label, summarize=SHOW_SUMMARIES)

    # split samples into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.34, random_state=42)

    for classifier_label in CLASSIFIERS:
        if classifier_label in EXCLUDE_CLASSIFIERS:
            continue
        # if classifier_label == 'k-means':
            # y_train[TARGET_CLASS] = label_encoder.transform(
            #     y_train[TARGET_CLASS])
            # y_test[TARGET_CLASS] = label_encoder.transform(
            #     y_test[TARGET_CLASS])

        classifier = CLASSIFIERS[classifier_label]
        pipeline = Pipeline(classifier['steps'])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classifier_label)

        # if clustering, no classification report!
        if classifier_label == 'k-means':
            print()
            # report_cluster(X_test, y_train, y_pred)
        else:
            print(classification_report(y_test, y_pred,
                                        target_names=['Non-VPN', 'VPN']))
        print()
