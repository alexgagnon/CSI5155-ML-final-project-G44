import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

logger = logging.getLogger(__name__)
log = logger.info


def sanitize_data(data, null_identifier=np.nan, null_replacement_value=None, remove_nulls=True):
    log("Rows before sanitization: {}".format(data.shape[0]))
    # need to handle that the dataset uses -1 as 'null'
    if (null_replacement_value != None):
        log('Replacing {} with {}'.format(
            null_identifier, null_replacement_value))
        data = data.replace(null_identifier, null_replacement_value)
    if (remove_nulls):
        nulls_before = data.isna().sum().sum()
        data = data.dropna(axis=0)
        nulls_after = data.isna().sum().sum()
        log("Removed {} nulls, {} left".format(
            nulls_before - nulls_after, nulls_after))
        data = data.reset_index(drop=True)
    log("Rows after sanitization: {}".format(data.shape[0]))
    return data


def print_metadata(df, label='', summarize=True):
    """Prints the pandas dataframe information"""
    log('\n')
    log(label)
    if (summarize):
        log(df.info())
    else:
        log('--- INFO ---')
        log(df.info())
        log('\n')
        log('--- DESCRIBE ---')
        log(df.describe())
        log('\n')
        log('--- HEAD ---')
        log(df.head())
        log('\n')


def print_eda(data, X, y, target_class, show_feature_distributions=True, show_plots_as_subplot=True, show_totals=True, show_correlations=True, show_box_plot=True, base_filename=''):
    # NOTE: plt.show(block=False) is async, need at least one plt.show()
    # to pause execution so you can read the graph

    # distribution of values in each feature
    if (show_feature_distributions):
        if (show_plots_as_subplot):
            fig, axes = plt.subplots(ncols=4, nrows=6, figsize=(15, 5))
            a = [i for i in axes for i in i]
            for i, ax in enumerate(a):
                sns.distplot(X[X.columns[i - 1]], ax=ax)
            plt.tight_layout()
            plt.savefig('results/' + base_filename +
                        'feature-distribution.png')
        else:
            for i in X.columns:
                plt.figure(figsize=(15, 5))
                sns.distplot(X[i])
                plt.savefig('results/' + base_filename +
                            'feature-distribution.png')

    if (show_box_plot):
        X.plot(kind='box')
        plt.savefig('results/' + base_filename + 'feature-boxplot.png')

    # totals for classes
    if (show_totals):
        cmap = sns.color_palette("Set2")
        sns.countplot(x=target_class, data=data, palette=cmap)
        plt.xticks(rotation=45)
        plt.savefig('results/' + base_filename + 'totals.png')

    # correlation of features
    if (show_correlations):
        corr = data[data.columns[1:]].corr()
        plt.figure(figsize=(15, 8))
        sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 20))
        plt.savefig('results/' + base_filename + 'feature-correlations.png')


def report_cluster(X_train, y_test, y_pred):
    correct = 0

    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    log(correct/len(X_train))


def print_cross_validation_results(results, file_name='result.txt', print_to_file=False):
    accuracies = results['accuracy']
    precisions = results['precision']
    recalls = results['recall']
    f1s = results['f1']
    table = [[fold_index + 1, accuracies[fold_index], precisions[fold_index], recalls[fold_index], f1s[fold_index]]
             for fold_index in range(len(accuracies))]

    table.append(['avg', accuracies.mean(), precisions.mean(),
                  recalls.mean(), f1s.mean()])
    table.append(['std', accuracies.std() * 2,
                  precisions.std() * 2, recalls.std() * 2, f1s.std() * 2])
    sheet = tabulate(
        table, headers=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1'])
    log(sheet)
    if (print_to_file):
        with open(file_name, 'w+') as writer:
            writer.write(sheet)
    log('\n')
