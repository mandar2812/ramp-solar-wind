from __future__ import division

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(), 
            GradientBoostingClassifier(
                learning_rate = 0.1, 
                n_estimators = 200, 
                subsample = 0.25))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
