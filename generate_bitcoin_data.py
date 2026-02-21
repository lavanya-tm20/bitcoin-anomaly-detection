import os
import numpy as np
import pandas as pd

class BitcoinDatasetGenerator:
    def __init__(self, output_dir='data', random_seed=42):
        self.output_dir = output_dir
        self.rng = np.random.RandomState(random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_normal(self, n):
        t = np.arange(n)
        price = 50000 + np.cumsum(self.rng.normal(0, 50, size=n))
        volume = np.abs(1000 + self.rng.normal(0, 200, size=n))
        tx_count = np.abs(500 + self.rng.normal(0, 50, size=n))
        fee = np.abs(0.0001 + self.rng.normal(0, 0.00005, size=n))
        difficulty = np.abs(1e12 + self.rng.normal(0, 1e9, size=n))
        spread = np.abs(self.rng.normal(0.5, 0.1, size=n))
        df = pd.DataFrame({
            'price': price,
            'volume': volume,
            'tx_count': tx_count,
            'fee': fee,
            'difficulty': difficulty,
            'spread': spread,
            'attack_type': ['normal'] * n,
            'is_anomaly': [0] * n
        })
        return df

    def generate_ddos(self, n):
        df = self.generate_normal(n)
        df['tx_count'] = df['tx_count'] * 5 + self.rng.normal(0, 10, size=n)
        df['attack_type'] = ['dos'] * n
        df['is_anomaly'] = [1] * n
        return df

    def generate_double_spend(self, n):
        df = self.generate_normal(n)
        df['fee'] = df['fee'] * 50 + self.rng.normal(0, 0.001, size=n)
        df['attack_type'] = ['double_spend'] * n
        df['is_anomaly'] = [1] * n
        return df

    def generate_51percent(self, n):
        df = self.generate_normal(n)
        df['difficulty'] = df['difficulty'] * 0.1 + self.rng.normal(0, 1e8, size=n)
        df['attack_type'] = ['51_percent'] * n
        df['is_anomaly'] = [1] * n
        return df

    def generate_complete_dataset(self, n_normal=850, n_ddos=30, n_double_spend=30, n_51percent=30, shuffle=True):
        parts = []
        parts.append(self.generate_normal(n_normal))
        parts.append(self.generate_ddos(n_ddos))
        parts.append(self.generate_double_spend(n_double_spend))
        parts.append(self.generate_51percent(n_51percent))
        df = pd.concat(parts, ignore_index=True)
        if shuffle:
            df = df.sample(frac=1, random_state=self.rng).reset_index(drop=True)
        out_path = os.path.join(self.output_dir, 'sample_data.csv')
        df.to_csv(out_path, index=False)
        return df
