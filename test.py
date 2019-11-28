import os
import shutil
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import interp, stats
from scipy.io.arff import loadarff
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
# from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.trees import HoeffdingTree
from util import print_metadata, print_eda, sanitize_data, print_cross_validation_results
from streamz.dataframe import DataFrame as StreamingDataFrame
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

CROSS_VALIDATION_FOLDS = 10
FILES = {
    '15s': 'TimeBasedFeatures-Dataset-15s-VPN.arff',
    #  '60s': 'TimeBasedFeatures-Dataset-60s-VPN.arff',
    # '120s': 'TimeBasedFeatures-Dataset-120s-VPN.arff'
    # '30s': 'TimeBasedFeatures-Dataset-30s-VPN.arff',
    # dropped due to inconsistent features compared to the other datasets
}
FILE_PATH_DIR = './datasets/Scenario A1/'
RFE_COLUMNS = None
SHOW_FEATURE_DESCRIPTIONS = False
SHOW_METADATA = False
SHOW_EDA = False
SHOW_ROC = False
PRINT_ROC = True
SHOW_CROSS_VALIDATION_RESULTS = True
PRINT_CROSS_VALIDATION_TO_FILE = True
SEED = 42
TARGET_CLASS = 'class1'
CLASSIFIERS = {
    'tree': DecisionTreeClassifier(random_state=SEED),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(n_estimators=10, random_state=SEED),
    'gb': GradientBoostingClassifier(random_state=SEED),
    'svm': SVC(random_state=SEED, gamma='scale')
    # 'voting': VotingClassifier()
    # 'kmeans': KMeans(),
    # 'hoeffding': HoeffdingTree()
}
OUTPUT_DIR = 'results'
NO_RFE = ['knn', 'svm']
FEATURE_SELECTION_DROP_COLUMNS = [
    'min_idle', 'mean_idle', 'max_idle', 'std_idle']
SCORING_METRIC = 'accuracy'

encoder = LabelEncoder()  # {LabelEncoder, OneHotEncoder, LabelBinarizer}
normalizer = Normalizer()  # {None, Normalizer, StandardScaler, RobustScaler}
# {None, 'RFE', 'manual'} 'PCA' was found to LOWER the accuracies
feature_selectors = [None, 'RFE', 'manual']
# pca = PCA(.80)

try:
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
except OSError:
    pass

# set up logging
logging.basicConfig(filename='results/output.log', format='',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False
log = logger.info

# store results for comparison
classifier_results = {}

# get stats on initial data
if (SHOW_EDA):
    for dataset_label, filename in FILES.items():
        dataset, meta = loadarff(FILE_PATH_DIR + filename)
        data = pd.DataFrame(dataset)
        # handle nulls, duplicates, etc.
        data = sanitize_data(data, -1, np.nan)

        # extract samples features and target feature from dataframe
        # analysis they aren't normally distributed
        X = pd.DataFrame(data.drop(TARGET_CLASS, axis=1))
        # we convert the target from categorical strings to labels 0 and 1
        y = pd.DataFrame(encoder.fit_transform(data[TARGET_CLASS]))
        print_eda(data, X, y, TARGET_CLASS)


for dataset_label, filename in FILES.items():
    # have to do this as DataFrame can't be deeply copied
    for classifier_name, classifier in CLASSIFIERS.items():

        # to be able to see which version of a classifier is best statistically
        classifier_versions = {}

        for feature_selector in feature_selectors:
            if (classifier_name in NO_RFE and feature_selector == 'RFE'):
                continue
            label = "{}-{}-{}".format(dataset_label,
                                      classifier_name, feature_selector)

            log(label)
            log('----------------------')
            dataset, meta = loadarff(FILE_PATH_DIR + filename)
            data = pd.DataFrame(dataset)

            # handle nulls, duplicates, etc.
            data = sanitize_data(data, -1, np.nan)

            # extract samples features and target feature from dataframe
            # analysis they aren't normally distributed
            X = pd.DataFrame(data.drop(TARGET_CLASS, axis=1))
            # we convert the target from categorical strings to labels 0 and 1
            y = pd.DataFrame(encoder.fit_transform(data[TARGET_CLASS]))

            columns = X.columns

            # we normalize the features as they contain very large values and from EDA
            if (normalizer != None):
                if (SHOW_FEATURE_DESCRIPTIONS):
                    log("\nBefore normalization:")
                    log(X.describe())
                X = pd.DataFrame(normalizer.fit_transform(X), columns=columns)
                if (SHOW_FEATURE_DESCRIPTIONS):
                    log("\nAfter normalization:")
                    log(X.describe())

            # DEPRECATED: PCA reduces the accuracies across the board
            # try to do feature selection to improve performance/accuracy
            # PCA doesn't depend on an estimator
            # NOTE: if using PCA, you will lose the column headers
            # if (feature_selector == 'PCA'):
            #     log('\n')
            #     log("Number of features before selection: {}".format(len(X.columns)))
            #     log(list(X.columns))
            #     X = pd.DataFrame(pca.fit_transform(X))
            #     log('\n')
            #     log("Number of features after selection: {}".format(len(X.columns)))
            #     log(list(X.columns))

            if (SHOW_METADATA):
                print_metadata(data, summarize=True)

            # split samples into training and test sets
            # MIGHT NOT NEED THIS
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state=SEED)

            # create validation folds
            cross_validation = StratifiedKFold(
                n_splits=CROSS_VALIDATION_FOLDS, random_state=SEED)

            # see if we can reduce features
            if (feature_selector != None):
                old_columns = X_train.columns
                new_columns = []
                log("\nNumber of features before selection: {}".format(
                    len(old_columns)))
                log(list(old_columns))
                # some classifiers do not have the means to rank, so can't use RFE
                if (feature_selector == 'RFE'):
                    classifier = RFE(
                        classifier, n_features_to_select=RFE_COLUMNS)
                    X_train = pd.DataFrame(classifier.fit_transform(
                        X_train, y_train.values.ravel()))
                    for i, val in enumerate(classifier.support_):
                        if (val == True):
                            new_columns.append(old_columns[i])

                elif (feature_selector == 'manual'):
                    X_train = X_train.drop(
                        FEATURE_SELECTION_DROP_COLUMNS, axis=1)
                    new_columns = X_train.columns

                log("\nNumber of features after selection: {}".format(
                    len(new_columns)))
                log(new_columns)

            accuracies = []
            precisions = []
            recalls = []
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for fold, split in enumerate(cross_validation.split(X_train, y_train)):
                fold_train_indexes, fold_test_indexes = split
                fold_X_train = X_train.iloc[fold_train_indexes]
                fold_y_train = y_train.iloc[fold_train_indexes]
                fold_X_test = X_train.iloc[fold_test_indexes]
                fold_y_test = y_train.iloc[fold_test_indexes]
                model = classifier.fit(
                    fold_X_train, fold_y_train.values.ravel())
                y_pred = model.predict(fold_X_test)

                accuracies.append(accuracy_score(fold_y_test, y_pred))
                precisions.append(precision_score(
                    fold_y_test, y_pred, average='binary'))
                recalls.append(recall_score(
                    fold_y_test, y_pred, average='binary'))

                # for ROC
                probabilities = model.predict_proba(fold_X_test)
                fpr, tpr, thresholds = roc_curve(
                    fold_y_test, probabilities[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

            results = {
                'accuracy': np.array(accuracies),
                'precision': np.array(precisions),
                'recall': np.array(recalls)
            }

            if (SHOW_CROSS_VALIDATION_RESULTS):
                feature_selector_name = feature_selector
                if (feature_selector == 'RFE'):
                    feature_selector_name += str(RFE_COLUMNS)
                file_name = "{}/{}-{}-{}.txt".format(
                    OUTPUT_DIR, dataset_label, classifier_name, feature_selector_name)
                print_cross_validation_results(
                    results, file_name=file_name, print_to_file=PRINT_CROSS_VALIDATION_TO_FILE)

            # code modified from sklearn site for computing ROC with cross
            # validation
            if (SHOW_ROC):
                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                         label='Chance', alpha=.8)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                plt.plot(mean_fpr, mean_tpr, color='b',
                         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (
                             mean_auc, std_auc),
                         lw=2, alpha=.8)

                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                                 color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.show()

            classifier_results[label] = results
            classifier_versions[feature_selector] = results

        log("Paired t-test comparison of feature selection methods")
        for a, b in list(combinations(classifier_versions.keys(), 2)):
            log("{} vs. {}".format(a, b))
            a = classifier_versions[a][SCORING_METRIC]
            b = classifier_versions[b][SCORING_METRIC]
            ttest = stats.ttest_rel(a, b)
            log("tvalue: {}, pvalue: {}".format(ttest.statistic, ttest.pvalue))

        log('\n')

# compute statistical significance of differences in models
if (len(classifier_results.keys()) > 1):
    log('Classifier evaluations')
    model_pairs = list(combinations(classifier_results.keys(), 2))
    model_triples = list(combinations(classifier_results.keys(), 3))

    for a, b in model_pairs:
        log("{} vs. {}".format(a, b))
        a = classifier_results[a][SCORING_METRIC]
        b = classifier_results[b][SCORING_METRIC]

        wilcoxon = stats.wilcoxon(a, b)
        log("Wilcoxons': {}, pvalue = {}".format(
            wilcoxon.statistic, wilcoxon.pvalue))
        log('\n')

    scores = list(
        map(lambda x: classifier_results[x][SCORING_METRIC], classifier_results.keys()))
    log(scores)
    # friedman = stats.friedmanchisquare()
    # log("Friedmans': {}".format(
    #     friedman.statistic, friedman.pvalue))
    # log('\n')
