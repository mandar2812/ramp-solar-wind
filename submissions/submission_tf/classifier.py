from __future__ import division

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import tensorflow as tf


class Classifier(BaseEstimator):
    def __init__(self):
        self.num_features = 12
        
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=0.01
        )
        

        self.layers = [
        tf.keras.layers.Dense(
            24, input_dim = self.num_features, kernel_initializer = 'normal', 
            activation = 'sigmoid', bias_regularizer = tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(
            8, kernel_initializer = 'normal', activation = 'sigmoid', 
            bias_regularizer = tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(2, activation = 'softmax')
        ]

        self.model = tf.keras.Sequential(self.layers)

        self.model.compile(
            optimizer = tf.keras.optimizers.RMSprop(), 
            loss = 'sparse_categorical_crossentropy')
        
    def fit(self, X, y):
        self.model.fit(X.values, y.values.astype(int), batch_size = 512, epochs = 40)

    def predict_proba(self, X):
        return self.model.predict_proba(X.values, batch_size = 512)
