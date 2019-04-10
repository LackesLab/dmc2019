# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb

sns.set(style="whitegrid")
# %%
df_train = pd.read_csv("../data/extended_train.csv", sep="|")


# %%
def score_function(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    f2_score = (0 if all(y_pred == 0) else metrics.fbeta_score(y_test, y_pred, beta=2))
    dmc_score = np.sum(confusion_matrix * np.array([[0, -25], [-5, 5]]))

    return accuracy, f2_score, dmc_score, confusion_matrix


# %%
df_cpy = df_train.copy()
scaler = StandardScaler()
df_cpy[['trustLevel', 'totalScanTimeInSeconds', 'grandTotal', 'lineItemVoids', 'scansWithoutRegistration',
        'quantityModifications', 'scannedLineItemsPerSecond', 'valuePerSecond', 'lineItemVoidsPerPosition',
        'totalScannedLineItems']] = scaler.fit_transform(df_cpy[['trustLevel', 'totalScanTimeInSeconds', 'grandTotal',
                                                                 'lineItemVoids', 'scansWithoutRegistration',
                                                                 'quantityModifications', 'scannedLineItemsPerSecond',
                                                                 'valuePerSecond', 'lineItemVoidsPerPosition',
                                                                 'totalScannedLineItems']])
y = df_cpy.fraud
X = df_cpy.drop(['fraud'], axis=1)
# %%

# %%
xgb_model2 = xgb.XGBClassifier(nthread=8, objective="binary:logistic", random_state=42, eval_metric="auc")
params = {
    "eta": uniform(0.1, 0.7),
    "gamma": uniform(0, 1),
    "learning_rate": uniform(0.03, 0.3),  # default 0.1
    "max_depth": randint(2, 8),  # default 3
    "n_estimators": randint(100, 400),  # default 100
    "subsample": uniform(0.6, 0.4),
    "lambda" : uniform(1e-3, 1),
"colsample_bytree" : uniform(0.1,0.9)
}


# %%
def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))


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


# %%
ftwo_scorer = metrics.make_scorer(metrics.fbeta_score, beta=3)
search = RandomizedSearchCV(xgb_model2, scoring=ftwo_scorer, param_distributions=params, random_state=42, n_iter=200,
                            cv=3, verbose=1, n_jobs=8, return_train_score=True)
search.fit(X, y)

report_best_scores(search.cv_results_, 3)
# %%

