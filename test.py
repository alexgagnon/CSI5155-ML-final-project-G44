from copy import deepcopy
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
# from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.trees import HoeffdingTree
from util import print_metadata, print_eda, sanitize_data, print_cross_validation_results

CROSS_VALIDATION_FOLDS = 10
FILES = {
    '15s': 'TimeBasedFeatures-Dataset-15s-VPN.arff',
    #  '60s': 'TimeBasedFeatures-Dataset-60s-VPN.arff',
    # '120s': 'TimeBasedFeatures-Dataset-120s-VPN.arff'
    # '30s': 'TimeBasedFeatures-Dataset-30s-VPN.arff',
    # dropped due to inconsistent features compared to the other datasets
}
FILE_PATH_DIR = './datasets/Scenario A1/'
SHOW_FEATURE_DESCRIPTIONS = True
SHOW_METADATA = False
SHOW_EDA = False
SHOW_ROC = False
SHOW_CROSS_VALIDATION_RESULTS = True
SEED = 42
TARGET_CLASS = 'class1'
CLASSIFIERS = {
    'tree': DecisionTreeClassifier(),
    'knn': KNeighborsClassifier(),
    # 'kmeans': KMeans()
}

encoder = LabelEncoder()  # {LabelEncoder, OneHotEncoder, LabelBinarizer}
normalizer = RobustScaler()  # {None, Normalizer, StandardScaler, RobustScaler}
feature_selector = None  # {None, 'PCA', 'RFE', 'normal-distribution'}
pca = PCA(.95)

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

    columns = X.columns

    # we normalize the features as they contain very large values and from EDA
    if (normalizer != None):
        if (SHOW_FEATURE_DESCRIPTIONS):
            print()
            print("Before normalization:")
            print(X.describe())
        X = pd.DataFrame(normalizer.fit_transform(X), columns=columns)
        if (SHOW_FEATURE_DESCRIPTIONS):
            print()
            print("After normalization:")
            print(X.describe())

    # try to do feature selection to improve performance/accuracy
    # PCA doesn't depend on an estimator
    # NOTE: if using PCA, you will lose the column headers
    if (feature_selector == 'PCA'):
        print()
        print("Number of features before selection: {}".format(len(X.columns)))
        print(list(X.columns))
        X = pd.DataFrame(pca.fit_transform(X))
        print()
        print("Number of features after selection: {}".format(len(X.columns)))
        print(list(X.columns))

    if (SHOW_METADATA):
        print_metadata(data, X, y, TARGET_CLASS, summarize=True)

    if (SHOW_EDA):
        print_eda(data, X, y, TARGET_CLASS)

    # split samples into training and test sets
    # MIGHT NOT NEED THIS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.34, random_state=SEED)

    # create validatio nfolds
    cross_validation = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS, random_state=SEED)

    classifier_results = {}

    for classifier_name, classifier in CLASSIFIERS.items():
        X_train = deepcopy(X_train)
        y_train = deepcopy(y_train)
        print(X_train.columns)
        if (feature_selector == 'RFE' or feature_selector == 'RFECV'):
            old_columns = X_train.columns
            classifier = RFE(classifier, n_features_to_select=5)
            print()
            print("Number of features before selection: {}".format(
                len(X_train.columns)))
            print(list(X_train.columns))
            X_train = pd.DataFrame(classifier.fit_transform(
                X_train, y_train.values.ravel()))
            print()
            new_columns = []
            for i, val in enumerate(classifier.support_):
                if (val == True):
                    new_columns.append(old_columns[i])
            print("Number of features after selection: {}".format(
                len(new_columns)))
            print(new_columns)

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
            model = classifier.fit(fold_X_train, fold_y_train.values.ravel())
            y_pred = model.predict(fold_X_test)

            accuracies.append(accuracy_score(fold_y_test, y_pred))
            precisions.append(precision_score(
                fold_y_test, y_pred, average='binary'))
            recalls.append(recall_score(
                fold_y_test, y_pred, average='binary'))

            if (SHOW_ROC):
                probabilities = model.predict_proba(fold_X_test)
                fpr, tpr, thresholds = roc_curve(
                    fold_y_test, probabilities[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot()
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='ROC fold {} (AUC = {:0.2f})'.format(fold, roc_auc))

        results = {
            'accuracies': np.array(accuracies),
            'precisions': np.array(precisions),
            'recalls': np.array(recalls)
        }

        if (SHOW_CROSS_VALIDATION_RESULTS):
            print(classifier_name)
            print_cross_validation_results(results)

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

        classifier_results[classifier_name] = results
        print()

    # compute statistical significance of differences in models
    if (len(CLASSIFIERS) > 1):
        print('Classifier evaluations')
        model_pairs = list(combinations(CLASSIFIERS.keys(), 2))
        model_triples = list(combinations(CLASSIFIERS.keys(), 3))

        for a, b in model_pairs:
            print("{} vs. {}".format(a, b))
            a = classifier_results[a]['accuracies']
            b = classifier_results[b]['accuracies']

            ttest = stats.ttest_rel(a, b)
            wilcoxon = stats.wilcoxon(a, b)
            print("Paired t-test: {}, pvalue = {}".format(ttest.statistic, ttest.pvalue))
            print("Wilcoxons': {}, pvalue = {}".format(
                wilcoxon.statistic, wilcoxon.pvalue))
            print()

        for a, b, c in model_triples:
            print("{} vs. {} vs. {}".format(a, b, c))
            a = classifier_results[a]['accuracies']
            b = classifier_results[b]['accuracies']
            c = classifier_results[c]['accuracies']

            friedman = stats.friedmanchisquare(a, b, c)
            print("Friedmans': {}".format(friedman.statistic, friedman.pvalue))
            print()
