import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_xgboost(X, y, save_path):
    # Convert labels so smallest becomes 0 and largest becomes 1
    # XGBoost requires labels 0 and 1
    import numpy as np
    unique = np.unique(y)

    if len(unique) > 2:
        print("Converting multi-class labels to binary...")
        y = (y != unique[0]).astype(int)   # First class = 0, others = 1

    print("Final classes used:", np.unique(y))

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    model.fit(X, y)

    # Save model
    joblib.dump(model, save_path + "_model.pkl")

    print("XGBoost trained and saved at:", save_path + "_model.pkl")
    return model


def predict_xgboost(model_path, X):
    model = joblib.load(model_path)
    return model.predict(X)
