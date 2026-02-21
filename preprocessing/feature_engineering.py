import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_cols = None
        self.label_col = None

    def fit_transform(self, df, label_col='attack_type', remove_outliers=True):
        """Preprocess the dataset and return cleaned df, features list, scaled X, encoded y."""

        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset.")

        self.label_col = label_col

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in numeric_cols:
            numeric_cols.remove(label_col)

        if len(numeric_cols) == 0:
            raise ValueError("Dataset has no numeric feature columns.")

        # Fill missing values safely
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Remove outliers using z-score
        if remove_outliers:
            z = np.abs((df[numeric_cols] - df[numeric_cols].mean()) /
                       (df[numeric_cols].std() + 1e-9))
            mask = (z < 3).all(axis=1)
            df = df[mask].reset_index(drop=True)

        # Scale numeric columns
        X_scaled = self.scaler.fit_transform(df[numeric_cols])

        # Encode labels fresh AFTER outlier removal
        y_raw = df[label_col].astype(str).values

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_raw)

        print("Final classes used:", np.unique(y_enc))

        self.feature_cols = numeric_cols

        return df, numeric_cols, X_scaled, y_enc

    def transform_new(self, df):
        """Transform new incoming data for predictions."""

        df = df.copy()

        # Ensure columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing feature '{col}' in new data.")

        df[self.feature_cols] = df[self.feature_cols].ffill().bfill()

        X_scaled = self.scaler.transform(df[self.feature_cols])

        return X_scaled
