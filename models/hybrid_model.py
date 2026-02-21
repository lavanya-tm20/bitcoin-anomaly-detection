import numpy as np
import joblib
import os

class HybridEngine:
    """
    Combine supervised probability confidence and anomaly scores into a risk score.
    risk = alpha*(1 - max_class_confidence) + (1-alpha)*normalized_anomaly_score
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha

    def compute_risk(self, supervised_proba, anomaly_score):
        # supervised_proba: NxC probabilities
        conf = supervised_proba.max(axis=1)  # higher => more confident (less risky)
        s = anomaly_score.astype(float)
        if s.max() == s.min():
            norm_anom = np.zeros_like(s)
        else:
            norm_anom = (s - s.min()) / (s.max() - s.min())
        risk = self.alpha * (1.0 - conf) + (1.0 - self.alpha) * norm_anom
        return np.clip(risk, 0.0, 1.0)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
