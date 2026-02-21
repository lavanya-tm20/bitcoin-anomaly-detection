import os
import pandas as pd
from generate_bitcoin_data import BitcoinDatasetGenerator
from pipeline import AnomalyDetectionPipeline

project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, 'sample_data.csv')
if not os.path.exists(csv_path):
    gen = BitcoinDatasetGenerator(output_dir=data_dir)
    gen.generate_complete_dataset(n_normal=810, n_ddos=30, n_double_spend=30, n_51percent=30)
df = pd.read_csv(csv_path)
pipeline = AnomalyDetectionPipeline(project_root)
processed, features, X_scaled, y_enc = pipeline.preprocess(df, label_col='attack_type')
pipeline.train_supervised(X_scaled, y_enc)
pipeline.train_unsupervised(X_scaled)
res = pipeline.infer(X_scaled)
print('Done. Sample risk scores (first 10):', res['risk_score'][:10])
