from sklearn.ensemble import RandomForestClassifier
import pickle

def train_random_forest(X, y, save_path):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )
    
    model.fit(X, y)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model
