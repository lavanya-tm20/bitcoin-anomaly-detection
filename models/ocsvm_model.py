from sklearn.svm import OneClassSVM
import numpy as np
import joblib
import os

class OCSVMWrapper:
    def __init__(self, nu=0.05, kernel='rbf'):
        self.nu = nu
        self.kernel = kernel
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel)

    def fit(self, X):
        self.model.fit(X)
        return self

    def scores(self, X):
        # higher score -> more anomalous (we invert decision_function to make larger = anomaly)
        return -self.model.decision_function(X).ravel()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
