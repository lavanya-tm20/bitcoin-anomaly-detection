import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report,
    silhouette_score
)
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from scipy.stats import zscore

# -------------------------
# Streamlit Page Setup
# -------------------------
st.set_page_config(
    page_title="Blockchain Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "Upload Dataset",
        "Dashboard",
        "Model Evaluation",
        "Feature Importance",
        "Attack Clusters",
        "Severity Analysis",
        "Anomaly Windows",
        "Download Output"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("📘 Final-Year Major Project")

# -------------------------
# Session State Defaults
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "numeric_cols" not in st.session_state:
    st.session_state.numeric_cols = None
if "scaled" not in st.session_state:
    st.session_state.scaled = None
if "synthetic_labels" not in st.session_state:
    st.session_state.synthetic_labels = None

# -------------------------
# Page: Upload Dataset
# -------------------------
if page == "Upload Dataset":
    st.title("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        st.session_state.df = df
        st.session_state.numeric_cols = numeric_cols

        st.success("Dataset successfully uploaded!")
        st.dataframe(df.head())

        if len(numeric_cols) < 2:
            st.error("Dataset must contain at least 2 numeric columns.")
            st.stop()

        # Synthetic anomaly labels (fallback if no true labels)
        z_scores = df[numeric_cols].apply(zscore)
        synthetic_labels = (z_scores.abs().mean(axis=1) > 2.5).astype(int)
        st.session_state.synthetic_labels = synthetic_labels

        # Normalization choice
        st.subheader("Normalization")
        norm = st.radio("Select normalization method:", ["StandardScaler", "Min-Max"])
        scaler = MinMaxScaler() if norm == "Min-Max" else StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols])
        st.session_state.scaled = scaled

        st.success("Preprocessing completed. Navigate to other pages.")
    else:
        st.info("Upload CSV to continue.")

# Stop here if dataset not uploaded yet
if st.session_state.df is None:
    st.stop()

# -------------------------
# Load objects from session
# -------------------------
df = st.session_state.df
numeric_cols = st.session_state.numeric_cols
scaled = st.session_state.scaled
synthetic_labels = st.session_state.synthetic_labels

# -------------------------
# ML Pipeline (original logic preserved)
# -------------------------
# Random Forest
rf = RandomForestClassifier(n_estimators=120, random_state=42)
rf.fit(scaled, synthetic_labels)
rf_pred = rf.predict_proba(scaled)[:, 1]

# XGBoost
xgb = XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric="logloss")
xgb.fit(scaled, synthetic_labels)
xgb_pred = xgb.predict_proba(scaled)[:, 1]

# KMeans (general)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled)

# One-Class SVM
ocsvm = OneClassSVM(kernel="rbf", nu=0.05)
svm_raw = ocsvm.fit_predict(scaled)
svm_pred = np.where(svm_raw == -1, 1, 0)

# Hybrid score & anomaly flag
df["hybrid_score"] = (
    0.35 * rf_pred +
    0.35 * xgb_pred +
    0.15 * clusters +
    0.15 * svm_pred
)
df["is_anomaly"] = (df["hybrid_score"] > df["hybrid_score"].mean() + df["hybrid_score"].std()).astype(int)

# PCA for visualization
pca = PCA(n_components=2)
pca_vals = pca.fit_transform(scaled)
df["pca1"] = pca_vals[:, 0]
df["pca2"] = pca_vals[:, 1]

# -------------------------
# Stage 2: KMeans on OCSVM outliers
# -------------------------
svm_indices = np.where(svm_pred == 1)[0]
svm_data = scaled[svm_pred == 1]

if len(svm_data) >= 2:
    kmeans_stage2 = KMeans(n_clusters=3, random_state=42)
    stage2_clusters = kmeans_stage2.fit_predict(svm_data)
    # Map back to df
    df["stage2_cluster"] = -1
    df.loc[svm_indices, "stage2_cluster"] = stage2_clusters
else:
    df["stage2_cluster"] = -1
    stage2_clusters = None

df["stage2_cluster"] = df["stage2_cluster"].astype(int)

# -------------------------
# Automatic cluster labeling (heuristic)
# -------------------------
cluster_label_map = {}
if stage2_clusters is not None:
    for c in sorted(np.unique(stage2_clusters)):
        idxs = svm_indices[stage2_clusters == c]
        if len(idxs) == 0:
            cluster_label_map[c] = "Unknown"
            continue
        centroid = df.loc[idxs, numeric_cols].mean()
        ddos_score = centroid.max()
        ds_score = centroid.median()
        pct51_score = centroid.min()
        scores = {"DDoS": ddos_score, "Double Spending": ds_score, "51% Vulnerability": pct51_score}
        cluster_label_map[c] = max(scores, key=scores.get)

def map_cluster_label(x):
    if x == -1:
        return "Normal"
    return cluster_label_map.get(x, "Unknown")

df["cluster_label"] = df["stage2_cluster"].apply(map_cluster_label)

# -------------------------
# Severity scoring
# -------------------------
hs_min, hs_max = df["hybrid_score"].min(), df["hybrid_score"].max()
df["severity_score"] = (df["hybrid_score"] - hs_min) / (hs_max - hs_min) if hs_max - hs_min > 0 else 0.0

def severity_label_fn(s):
    if s < 0.33:
        return "Low"
    if s < 0.66:
        return "Medium"
    return "High"

df["severity_label"] = df["severity_score"].apply(severity_label_fn)

# -------------------------
# Time-series anomaly windows
# -------------------------
anomaly_idx = df.index[df["is_anomaly"] == 1].tolist()
windows = []
gap_threshold = 3

if len(anomaly_idx) > 0:
    start = anomaly_idx[0]
    end = anomaly_idx[0]
    for i in anomaly_idx[1:]:
        if i - end <= gap_threshold:
            end = i
        else:
            windows.append((start, end))
            start = i
            end = i
    windows.append((start, end))

# -------------------------
# Pages
# -------------------------
ieee_colors = {
    "blue": "#0072BD",
    "red": "#D95319",
    "green": "#77AC30",
    "orange": "#EDB120",
    "purple": "#7E2F8E",
    "cyan": "#4DBEEE",
    "magenta": "#A2142F",
    "yellow": "#FFED6F",
    "gray": "#7F7F7F"
}

# Dashboard
if page == "Dashboard":
    st.title("📊 Anomaly Detection Dashboard")

    acc = accuracy_score(synthetic_labels, df["is_anomaly"])
    st.subheader("📌 Model Performance Summary")
    st.metric("Hybrid Model Accuracy", f"{acc*100:.2f}%")

    cm = confusion_matrix(synthetic_labels, df["is_anomaly"])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        cm2 = np.zeros((2, 2), dtype=int)
        cm2[:cm.shape[0], :cm.shape[1]] = cm
        tn, fp, fn, tp = cm2.ravel()

    cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Anomaly"], columns=["Pred Normal", "Pred Anomaly"])
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale=[ieee_colors["blue"], ieee_colors["red"]], title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Positives (TP)", int(tp))
    col2.metric("True Negatives (TN)", int(tn))
    col3.metric("False Positives (FP)", int(fp))
    col4.metric("False Negatives (FN)", int(fn))

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Anomaly Distribution")
        fig_pie = px.pie(df, names="is_anomaly", color="is_anomaly", title="Anomaly vs Normal",
                         color_discrete_map={0: ieee_colors["blue"], 1: ieee_colors["red"]})
        st.plotly_chart(fig_pie, use_container_width=True)
    with right:
        st.subheader("PCA Scatter")
        fig_pca = px.scatter(df, x="pca1", y="pca2", color="is_anomaly",
                             color_discrete_map={0: ieee_colors["blue"], 1: ieee_colors["red"]},
                             title="PCA: Anomalies Highlighted")
        st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Hybrid Score Time-series (Anomalies highlighted)")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(y=df["hybrid_score"], mode="lines", name="Hybrid Score", line=dict(color=ieee_colors["blue"])))
    fig_ts.add_trace(go.Scatter(x=df.index[df["is_anomaly"] == 1],
                                y=df["hybrid_score"][df["is_anomaly"] == 1],
                                mode="markers", marker=dict(color=ieee_colors["red"], size=7), name="Anomalies"))
    st.plotly_chart(fig_ts, use_container_width=True)

# Model Evaluation Page
if page == "Model Evaluation":
    st.title("📏 Model Evaluation")
    acc = accuracy_score(synthetic_labels, df["is_anomaly"])
    cm = confusion_matrix(synthetic_labels, df["is_anomaly"])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        cm2 = np.zeros((2, 2), dtype=int)
        cm2[:cm.shape[0], :cm.shape[1]] = cm
        tn, fp, fn, tp = cm2.ravel()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("TP", int(tp))
    col3.metric("TN", int(tn))
    col4.metric("FP", int(fp))
    col5, col6 = st.columns(2)
    col5.metric("FN", int(fn))
    col6.write("")

    st.subheader("Classification Report")
    st.code(classification_report(synthetic_labels, df["is_anomaly"]), language="txt")

    fpr, tpr, _ = roc_curve(synthetic_labels, df["hybrid_score"])
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})", line=dict(color=ieee_colors["blue"])))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=ieee_colors["gray"], dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    precision, recall, _ = precision_recall_curve(synthetic_labels, df["hybrid_score"])
    pr_auc = auc(recall, precision)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AUC={pr_auc:.3f})", line=dict(color=ieee_colors["red"])))
    fig_pr.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True)

# Feature Importance
if page == "Feature Importance":
    st.title("🔥 Feature Importance (Random Forest)")
    fi = pd.DataFrame({
        "feature": numeric_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=True)
    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importance",
                    color="importance", color_continuous_scale=[ieee_colors["blue"], ieee_colors["red"]])
    st.plotly_chart(fig_fi, use_container_width=True)

# Attack Clusters
if page == "Attack Clusters":
    st.title("🎯 Attack Clusters")
    st.subheader("Cluster label distribution")
    st.bar_chart(df["cluster_label"].value_counts())

    st.subheader("Clustered anomalies (PCA)")
    if df[df["is_anomaly"] == 1].shape[0] > 0:
        fig_clusters = px.scatter(df[df["is_anomaly"] == 1], x="pca1", y="pca2", color="cluster_label",
                                  color_discrete_map={k: v for k, v in zip(df["cluster_label"].unique(), list(ieee_colors.values()))},
                                  title="Anomalies by Cluster Label (PCA)")
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.info("No anomalies to display clusters.")

# Severity Analysis
if page == "Severity Analysis":
    st.title("⚖️ Severity Analysis")
    left, right = st.columns(2)
    with left:
        st.subheader("Severity distribution (anomalies only)")
        if df[df["is_anomaly"] == 1].shape[0] > 0:
            st.bar_chart(df[df["is_anomaly"] == 1]["severity_label"].value_counts())
        else:
            st.info("No anomalies to show severity.")
    with right:
        st.subheader("Severity by Cluster")
        if df[df["is_anomaly"] == 1].shape[0] > 0:
            fig_box = px.box(df[df["is_anomaly"] == 1], x="cluster_label", y="severity_score", color="cluster_label",
                             color_discrete_map={k: v for k, v in zip(df["cluster_label"].unique(), list(ieee_colors.values()))},
                             title="Severity Score per Cluster")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No anomalies to show.")

    st.subheader("Top anomalies (by severity)")
    top_anoms = df[df["is_anomaly"] == 1].sort_values("severity_score", ascending=False).head(20)
    if not top_anoms.empty:
        st.dataframe(top_anoms[[*numeric_cols[:6], "hybrid_score", "severity_score", "severity_label", "cluster_label"]].head(20))
    else:
        st.info("No anomalies detected.")

# Anomaly Windows
if page == "Anomaly Windows":
    st.title("🕒 Anomaly Windows (Episodes)")
    if len(windows) == 0:
        st.info("No anomaly windows detected.")
    else:
        rows = []
        for i, (s, e) in enumerate(windows, 1):
            win = df.loc[s:e]
            predominant = win["cluster_label"].mode().iloc[0] if win["cluster_label"].mode().size > 0 else "N/A"
            rows.append({
                "Window": i,
                "Start": int(s),
                "End": int(e),
                "Length": int(e - s + 1),
                "Anomaly Count": int(win["is_anomaly"].sum()),
                "Avg Severity": round(float(win["severity_score"].mean()), 4),
                "Predominant Attack": predominant
            })
        st.table(pd.DataFrame(rows))

# Download Output
if page == "Download Output":
    st.title("⬇ Download Enriched Results")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "anomaly_results_enriched.csv",
        "text/csv"
    )
    st.success("Downloaded file contains hybrid_score, is_anomaly, stage2_cluster, cluster_label, severity_score, severity_label, pca1, pca2.")
