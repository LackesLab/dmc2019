import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split


def calc_scores(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    f2_score = (0 if all(y_pred == 0) else metrics.fbeta_score(y_test, y_pred, beta=2))
    dmc_score = np.sum(confusion_matrix * np.array([[0, -25], [-5, 5]]))

    return accuracy, f2_score, dmc_score, confusion_matrix


def own_scorer(estimator, X_val, ground_truth):
    prediction = estimator.predict(X_val)
    confusion_matrix = metrics.confusion_matrix(ground_truth, prediction)
    dmc_score = np.sum(confusion_matrix * np.array([[0, -25], [-5, 5]]))
    return dmc_score


def own_f2_score(estimator, X_val, ground_truth):
    prediction = estimator.predict(X_val)
    return 0 if all(prediction == 0) else metrics.fbeta_score(ground_truth, prediction, beta=2)


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
