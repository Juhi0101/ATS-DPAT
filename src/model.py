# =========================================
# ADVANCED ML PIPELINE (RESEARCH-GRADE)
# =========================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------
# CONFIG
# ------------------------------

DATA_PATH = "../data/processed/final_dataset.csv"
OUTPUT_DIR = "../outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------
# LOAD DATA
# ------------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df


# ------------------------------
# PREPROCESS
# ------------------------------

def preprocess(df):
    non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if "score" in non_numeric_cols:
        non_numeric_cols.remove("score")

    print(f"🧹 Dropping: {non_numeric_cols}")

    df = df.drop(columns=non_numeric_cols, errors="ignore")
    df = df.fillna(0)

    return df


# ------------------------------
# CORRELATION HEATMAP
# ------------------------------

def plot_correlation(df):
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()

    print("✅ Correlation heatmap saved")


# ------------------------------
# SPLIT DATA
# ------------------------------

def split_data(df):
    X = df.drop(columns=["score"])
    y = df["score"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------------------
# TRAIN MODELS
# ------------------------------

def train_models(X_train, y_train):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


# ------------------------------
# EVALUATION (WITH CV)
# ------------------------------

def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []

    for name, model in models.items():
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        results.append({
            "model": name,
            "MAE": round(mae, 3),
            "R2": round(r2, 3),
            "CV_R2_mean": round(cv_scores.mean(), 3)
        })

        print(f"\n{name}")
        print(f"MAE: {mae:.3f}")
        print(f"R2: {r2:.3f}")
        print(f"CV R2: {cv_scores.mean():.3f}")

    return pd.DataFrame(results)


# ------------------------------
# FEATURE IMPORTANCE
# ------------------------------

def plot_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns

        sorted_idx = np.argsort(importances)[::-1]

        top_n = min(10, len(importances))

        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[sorted_idx][:top_n])
        plt.xticks(range(top_n), feature_names[sorted_idx][:top_n], rotation=45)

        plt.title("Top Feature Importance (Random Forest)")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300)
        plt.close()

        print("✅ Feature importance saved")


# ------------------------------
# SIMPLE INTERPRETABILITY
# ------------------------------

def print_top_features(model, X):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns

        sorted_idx = np.argsort(importances)[::-1]

        print("\n🔝 Top Influential Features:")
        for i in sorted_idx[:10]:
            print(f"{feature_names[i]}: {importances[i]:.4f}")


# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():
    print("🔹 Loading data...")
    df = load_data(DATA_PATH)

    print("🔹 Preprocessing...")
    df = preprocess(df)

    print("🔹 Correlation analysis...")
    plot_correlation(df)

    print("🔹 Splitting...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("🔹 Training...")
    models = train_models(X_train, y_train)

    print("🔹 Evaluating...")
    results = evaluate_models(models, X_train, y_train, X_test, y_test)

    print("\n📊 Model Comparison:")
    print(results)

    print("🔹 Feature importance...")
    rf_model = models["RandomForest"]
    plot_feature_importance(rf_model, X_train)
    print_top_features(rf_model, X_train)


if __name__ == "__main__":
    main()