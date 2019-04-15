# ********************************************************************************;
#  _____              __          __   _            __  __             
# |  __ \             \ \        / /  | |          |  \/  |            
# | |  | | ___  ___ _ _\ \  /\  / /_ _| |_ ___ _ __| \  / | ___  _ __  
# | |  | |/ _ \/ _ \ '_ \ \/  \/ / _` | __/ _ \ '__| |\/| |/ _ \| '_ \ 
# | |__| |  __/  __/ |_) \  /\  / (_| | ||  __/ |  | |  | | (_) | | | |
# |_____/ \___|\___| .__/ \/  \/ \__,_|\__\___|_|  |_|  |_|\___/|_| |_|
#                  | |                                                 
#                  |_|    
#
# Project           : Master thesis - DeepWaterMon
# Program name      : random_forest_multiple.py
# School            : HEIA-FR
# Author            : Lucien Aymon
# Date created      : 07.12.2018
# Purpose           : Apply Random Forest on dataset, show and compute results
#                           Based on Scikit-Learn examples and F. Carrino project
# Revision History  :
# Date        Author      Ref    Revision
# 
# Input: Path of the dataset (str)
# Output: Data summary and Random Forest results, shown on the console
# ********************************************************************************;

# ------------------------------------
# Import
# ------------------------------------
import datetime
import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from pushbullet import Pushbullet
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import RobustScaler, StandardScaler

# -----------------------------------------
# Global constats and variables
# -----------------------------------------
NOTIFICATION_ON = False
PUSHBULLET_API_KEY = "UseYourOwnPushBulletAPIKey"
MULTIPLE_FILES = 0
dataset_path = sys.argv[1]
GRID_SEARCH = 1

# -----------------------------------------
# Configuration for mobile notification
# -----------------------------------------
pb = Pushbullet(PUSHBULLET_API_KEY)

# -----------------------------------------
# ID conversion, number to bit position
# -----------------------------------------
def bin_to_id(bin):
    try:
        return bin.index('1')
    except:
        print("Error with the simulation results")
        return 0

# -----------------------------------------
# Utility function to report best scores
# -----------------------------------------
def report(results, n_top=3):
    params = {}
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            if i is 1:
                params = results['params'][candidate]
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return params

# -----------------------------------------
# Random search function
# -----------------------------------------
def random_search_rf(X, y, classifier, max_features):
    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                "max_features": None,
                "min_samples_split": sp_randint(2, 11),
                "bootstrap": [True, False]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(classifier, param_distributions=param_dist,
                                    n_iter=n_iter_search, cv=5)

    start = time.time()
    random_search.fit(X, y)
    msg_notif = "RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." % ((time.time() - start), n_iter_search)
    print(msg_notif)
        
    # Send of the notification
    push = pb.push_note("WaterMon RandomSearch", msg_notif)

    return report(random_search.cv_results_)

# -----------------------------------------
# Grid search function
# -----------------------------------------
def grid_search_rf(X, y, classifier, max_features):
    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                "min_samples_split": [2, 3, 10],
                "bootstrap": [True, False]}

    # run grid search
    grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=5)
    start = time.time()
    grid_search.fit(X, y)
    msg_notif = "GridSearchCV took %.2f seconds" % ((time.time() - start))
    print(msg_notif)
    # Sending the notification
    push = pb.push_note("WaterMon GridSearch", msg_notif)
    return report(grid_search.cv_results_)
def main():
    # -----------------------------------------
    # Import the dataset and print details
    # -----------------------------------------
    print("Importing dataset...")
    dataset = pd.read_csv(dataset_path, error_bad_lines=False, dtype={'c': object})
    print("Dataset imported")

    # If the dataset is explode in multiple file, number of file is specified on global constantes
    if MULTIPLE_FILES is not 0:
        dataset_multiple = []
        dataset_multiple.append(dataset)
        path = os.path.splitext(dataset_path)[0]
        for i in range(1, MULTIPLE_FILES):
            path_i = "{0}_{1}.csv".format(path, i)
            print(path_i)
            dataset_multiple.append(pd.read_csv(path_i, error_bad_lines=False))
            dataset = pd.concat(dataset_multiple, ignore_index=True)
    push = pb.push_note("WaterMon Dataset", "Dataset loaded")

    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby('c').size())

    # -----------------------------------------
    # Explode and modify features and labels
    # -----------------------------------------
    x = dataset.filter(like='p_').values
    nbr_features = x.shape[1]
    y = dataset.filter(like='d_').values
    x = StandardScaler().fit_transform(x)

    first_field = dataset.filter(like='p_').columns[0]
    last_field = dataset.filter(like='p_').columns[len(dataset.filter(like='p_1').columns)-1]
    X = dataset.loc[:, first_field:last_field]
    Y = dataset.loc[:, 'c']

    # -----------------------------------------
    # Split into train and test sets
    # -----------------------------------------
    test_size = 0.20
    seed = 46
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # -----------------------------------------
    # Scaling the data
    # -----------------------------------------
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # ---------------------------------------------------
    # Random Forest classifier and hyperparametrisation
    # ---------------------------------------------------
    clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)

    # max_features is 11 if there is more than 11 different class
    print("Number of features (pressure): {0}".format(nbr_features))
    if nbr_features > 11:
        max_features = 11
    else:
        max_features = nbr_features-1

    if GRID_SEARCH is 1:
        params = grid_search_rf(X_train, Y_train, clf, max_features)
    else:
        print(max_features)
        params = grid_search_rf(X_train, Y_train, clf, max_features)

    classifier = RandomForestClassifier(bootstrap=params['bootstrap'], max_depth=params['max_depth'], max_features=None, min_samples_leaf=1, min_samples_split=params['min_samples_split'], n_estimators=20, n_jobs=-1)

    # ---------------------------------------------------------------
    # Cross-validation with K-Fold and printing learning curve
    # ---------------------------------------------------------------
    cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)

    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train_scaled, Y_train.values.ravel(), cv=cv)
    push = pb.push_note("WaterMon Learning curve", "Learning curve computed")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    plt.legend(loc="best")
    plt.show(block = False)

    # ---------------------------------------------------------------
    # Confusion matrix on the test set and importances graphs
    # ---------------------------------------------------------------
    classifier.fit(X_train_scaled, Y_train.values.ravel())

    importances = classifier.feature_importances_
    feature_importances = pd.DataFrame(classifier.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
    print("Features importances: {0}".format(feature_importances))

    feature_importances.plot.bar()  # All importances
    plt.title('Features importance')
    plt.xlabel('Junctions')
    plt.ylabel('Importance')
    plt.show(block = False)

    feature_importances.head(10).plot.bar() # 10 most important features
    plt.title('10 most important features')
    plt.xlabel('Junctions')
    plt.ylabel('Importance')
    plt.show(block = False)

    feature_importances.tail(10).plot.bar() # 10 less important features
    plt.title('10 less important features')
    plt.xlabel('Junctions')
    plt.ylabel('Importance')
    plt.show(block = False)

    feature_importances.sample(n=10, replace=True, random_state=1).plot.bar()  # 10 features with random importances
    plt.title('10 random features')
    plt.xlabel('Junctions')
    plt.ylabel('Importance')
    plt.show(block = False)
    y_pred = classifier.predict(X_test_scaled)

    # Confusion matrix
    cnf_matrix = confusion_matrix(Y_test, y_pred)
    labels = list(set(dataset['c']))
    labels = [bin_to_id(x) for x in labels]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cnf_matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show(block = False)

    normalize = False
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cnf_matrix)

    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    plt.figure()
    class_names = list(set(dataset['c']))
    class_names = [bin_to_id(x) for x in class_names]
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                        title='Normalized confusion matrix')
    plt.show()

    push = pb.push_note("WaterMon RF", "accuracy_score: " + str(accuracy_score(Y_test, y_pred)))

    # ---------------------------------------------------------------
    # Accuracy and metrics printing
    # ---------------------------------------------------------------

    print("Metrics per classes")
    print("accuracy_score: " + str(accuracy_score(Y_test, y_pred)))
    print("precision_score: " + str(precision_score(Y_test, y_pred, average=None)))
    print("recall_score: " + str(recall_score(Y_test, y_pred, average=None)))
    print("f1_score: " + str(f1_score(Y_test, y_pred, average=None)))

    print("Metrics (average)")
    print("accuracy_score: " + str(accuracy_score(Y_test, y_pred)))
    print("precision_score: " + str(precision_score(Y_test, y_pred, average="macro")))
    print("recall_score: " + str(recall_score(Y_test, y_pred, average="macro")))
    print("f1_score: " + str(f1_score(Y_test, y_pred, average="macro")))

if __name__ == "__main__":
    main()