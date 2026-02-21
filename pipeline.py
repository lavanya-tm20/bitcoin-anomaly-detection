import os, json
import numpy as np
import pandas as pd
from preprocessing.feature_engineering import FeatureEngineer
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.ocsvm_model import OCSVMWrapper
from models.kmeans_model import KMeansAnomaly
from models.hybrid_model import HybridEngine
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import silhouette_score

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bitcoin_blockchain_data.csv")  # default
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
EVAL_PATH = os.path.join(MODEL_DIR, "evaluation.json")

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Preprocessing...")
    fe = FeatureEngineer()
    processed_df, feature_cols, X_scaled, y_enc = fe.fit_transform(df, label_col='attack_type')

    print(f"Features: {feature_cols}")
    print(f"Samples after outlier removal: {X_scaled.shape[0]}")

    # Train supervised models
    rf_path = os.path.join(MODEL_DIR, "rf.pkl")
    xgb_path = os.path.join(MODEL_DIR, "xgb.pkl")
    print("Training RandomForest...")
    rf = train_random_forest(X_scaled, y_enc, rf_path)
    print("Training XGBoost...")
    xgb = train_xgboost(X_scaled, y_enc, xgb_path)

    # Supervised predictions for evaluation
    print("Generating supervised predictions for evaluation...")
    rf_pred = rf.predict(X_scaled)
    xgb_pred = xgb.predict(X_scaled)
    # average predicted probabilities for hybrid
    rf_proba = rf.predict_proba(X_scaled)
    xgb_proba = xgb.predict_proba(X_scaled)
    sup_proba = (rf_proba + xgb_proba) / 2.0
    sup_pred = np.argmax(sup_proba, axis=1)

    # Unsupervised models
    print("Training OCSVM...")
    ocsvm = OCSVMWrapper(nu=0.05)
    ocsvm.fit(X_scaled)
    ocsvm.save(os.path.join(MODEL_DIR, "ocsvm.pkl"))

    print("Training KMeans...")
    kmeans = KMeansAnomaly(n_clusters=3)
    kmeans.fit(X_scaled)
    kmeans.save(os.path.join(MODEL_DIR, "kmeans.pkl"))

    # Anomaly scores
    oc_scores = ocsvm.scores(X_scaled)
    km_dists = kmeans.distances(X_scaled)
    combined_anom = ( (oc_scores - oc_scores.min()) / (oc_scores.max() - oc_scores.min() + 1e-9)
                    + (km_dists - km_dists.min()) / (km_dists.max() - km_dists.min() + 1e-9) ) / 2.0

    # Hybrid risk
    hybrid = HybridEngine(alpha=0.6)
    risk_scores = hybrid.compute_risk(sup_proba, combined_anom)
    hybrid.save(os.path.join(MODEL_DIR, "hybrid.pkl"))

    # Evaluate supervised (use sup_pred vs y_enc)
    acc = accuracy_score(y_enc, sup_pred)
    prec = precision_score(y_enc, sup_pred, average='weighted', zero_division=0)
    rec = recall_score(y_enc, sup_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_enc, sup_pred, average='weighted', zero_division=0)
    cr = classification_report(y_enc, sup_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_enc, sup_pred).tolist()

    # KMeans silhouette
    try:
        silhouette = float(silhouette_score(X_scaled, kmeans.model.labels_))
    except Exception:
        silhouette = None

    evaluation = {
        "sample_count": int(X_scaled.shape[0]),
        "feature_count": len(feature_cols),
        "supervised": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "classification_report": cr,
            "confusion_matrix": cm
        },
        "unsupervised": {
            "ocsvm_anomaly_stats": {
                "min": float(np.min(oc_scores)),
                "max": float(np.max(oc_scores)),
                "mean": float(np.mean(oc_scores))
            },
            "kmeans_silhouette": silhouette
        },
        "hybrid": {
            "risk_score_stats": {
                "min": float(risk_scores.min()),
                "max": float(risk_scores.max()),
                "mean": float(risk_scores.mean())
            }
        }
    }

    save_json(evaluation, EVAL_PATH)
    print("Training and evaluation complete. Evaluation saved to:", EVAL_PATH)

    # Save processed data with predictions & scores for later use
    out = processed_df.reset_index(drop=True).copy()
    out['sup_pred'] = sup_pred
    out['sup_confidence'] = sup_proba.max(axis=1)
    out['ocsvm_score'] = oc_scores
    out['kmeans_distance'] = km_dists
    out['anomaly_score'] = combined_anom
    out['risk_score'] = risk_scores
    processed_out_path = os.path.join(PROJECT_ROOT, "data", "processed_data.csv")
    out.to_csv(processed_out_path, index=False)
    print("Processed data + scores saved to:", processed_out_path)

if __name__ == "__main__":
    main()
