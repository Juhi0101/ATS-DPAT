# ==============================
# ADVANCED ANALYSIS PIPELINE
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------

DATA_PATH = "../data/raw/data.csv"
OUTPUT_DIR = "../outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------
# LOAD & VALIDATE DATA
# ------------------------------

def load_data(path):
    df = pd.read_csv(path)

    required_cols = {"resume_id", "version", "tool", "score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    return df


# ------------------------------
# CONSISTENCY ANALYSIS (RQ1)
# ------------------------------

def compute_consistency(df):
    grouped = df.groupby(["resume_id", "version"])["score"]

    consistency = grouped.agg(
        max_score="max",
        min_score="min",
        mean_score="mean",
        std_dev="std",
        variance="var"
    ).reset_index()

    consistency["range"] = consistency["max_score"] - consistency["min_score"]

    return consistency


# ------------------------------
# TOOL-LEVEL CONSISTENCY
# ------------------------------

def tool_level_variation(df):
    tool_stats = df.groupby("tool")["score"].agg(
        mean="mean",
        std="std",
        variance="var"
    ).reset_index()

    return tool_stats


# ------------------------------
# SENSITIVITY ANALYSIS (RQ2)
# ------------------------------

def compute_sensitivity(df):
    pivot = df.pivot_table(
        index=["resume_id", "tool"],
        columns="version",
        values="score"
    ).reset_index()

    # Handle missing versions safely
    for col in ["A", "B", "C"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["change_B"] = pivot["B"] - pivot["A"]
    pivot["change_C"] = pivot["C"] - pivot["A"]

    pivot["abs_change_B"] = pivot["change_B"].abs()
    pivot["abs_change_C"] = pivot["change_C"].abs()

    return pivot


# ------------------------------
# PLOTTING
# ------------------------------

def plot_consistency(consistency):
    plt.figure(figsize=(12, 6))
    plt.bar(
        consistency["resume_id"] + "_" + consistency["version"],
        consistency["range"]
    )
    plt.xticks(rotation=45)
    plt.title("ATS Consistency (Score Range Across Tools)")
    plt.xlabel("Resume-Version")
    plt.ylabel("Score Range")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "consistency_range.png"), dpi=300)
    plt.close()


def plot_sensitivity(sensitivity):
    plt.figure(figsize=(12, 6))
    sensitivity[["abs_change_B", "abs_change_C"]].mean().plot(kind="bar")
    plt.title("Average Sensitivity Across Resume Variations")
    plt.ylabel("Average Absolute Score Change")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_avg.png"), dpi=300)
    plt.close()


def plot_tool_variation(tool_stats):
    plt.figure(figsize=(10, 5))
    plt.bar(tool_stats["tool"], tool_stats["std"])
    plt.title("Tool-wise Score Variability (Std Dev)")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tool_variability.png"), dpi=300)
    plt.close()


# ------------------------------
# SAVE RESULTS
# ------------------------------

def save_outputs(consistency, sensitivity, tool_stats):
    consistency.to_csv(os.path.join(OUTPUT_DIR, "consistency.csv"), index=False)
    sensitivity.to_csv(os.path.join(OUTPUT_DIR, "sensitivity.csv"), index=False)
    tool_stats.to_csv(os.path.join(OUTPUT_DIR, "tool_stats.csv"), index=False)


# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():
    df = load_data(DATA_PATH)

    consistency = compute_consistency(df)
    sensitivity = compute_sensitivity(df)
    tool_stats = tool_level_variation(df)

    save_outputs(consistency, sensitivity, tool_stats)

    plot_consistency(consistency)
    plot_sensitivity(sensitivity)
    plot_tool_variation(tool_stats)

    print("Analysis complete. Outputs saved.")


if __name__ == "__main__":
    main()