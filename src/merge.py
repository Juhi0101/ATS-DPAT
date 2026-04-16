# =========================================
# DATA MERGING PIPELINE (FIXED VERSION)
# =========================================

import os
import pandas as pd
from similarity import ats_match

# ------------------------------
# CONFIG
# ------------------------------

DATA_PATH = "../data/raw/data.csv"
FEATURES_PATH = "../data/processed/features.csv"
RESUME_FOLDER = "../data/resumes/"
JD_FOLDER = "../data/jd/"

OUTPUT_PATH = "../data/processed/final_dataset.csv"

os.makedirs("../data/processed", exist_ok=True)


# ------------------------------
# LOAD DATA
# ------------------------------

def load_data():
    ats_df = pd.read_csv(DATA_PATH)
    features_df = pd.read_csv(FEATURES_PATH)

    # Normalize resume_id format → r1, r2...
    ats_df["resume_id"] = ats_df["resume_id"].str.lower().str.replace("r0", "r")
    features_df["resume_id"] = features_df["resume_id"].str.lower().str.replace("r0", "r")

    return ats_df, features_df


# ------------------------------
# LOAD TEXT FILE
# ------------------------------

def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------
# MAP RESUME → JD
# ------------------------------

def get_jd_filename(resume_id):
    # r1 → jd1.txt
    num = int(resume_id.replace("r", ""))
    return f"jd{num}.txt"


# ------------------------------
# GENERATE SIMILARITY DATA
# ------------------------------

def compute_similarity_features(features_df):
    results = []

    for _, row in features_df.iterrows():
        resume_id = row["resume_id"]   # r1
        version = row["version"]       # A

        resume_file = f"{resume_id}_{version}.txt"
        resume_path = os.path.join(RESUME_FOLDER, resume_file)

        jd_file = get_jd_filename(resume_id)
        jd_path = os.path.join(JD_FOLDER, jd_file)

        if not os.path.exists(resume_path):
            print(f"❌ Missing resume: {resume_file}")
            continue

        if not os.path.exists(jd_path):
            print(f"❌ Missing JD: {jd_file}")
            continue

        resume_text = load_text(resume_path)
        job_text = load_text(jd_path)

        ats_result = ats_match(resume_text, job_text)

        results.append({
            "resume_id": resume_id,
            "version": version,
            "similarity_score": ats_result["similarity_score"],
            "final_score_predicted": ats_result["final_score"],
            "num_matched_skills": len(ats_result["matched_skills"]),
            "num_missing_skills": len(ats_result["missing_skills"]),
            "num_matched_keywords": len(ats_result["matched_keywords"]),
            "num_missing_keywords": len(ats_result["missing_keywords"])
        })

    return pd.DataFrame(results)


# ------------------------------
# MERGE EVERYTHING
# ------------------------------

def merge_all(ats_df, features_df, similarity_df):
    merged = pd.merge(
        ats_df,
        features_df,
        on=["resume_id", "version"],
        how="inner"
    )

    final_df = pd.merge(
        merged,
        similarity_df,
        on=["resume_id", "version"],
        how="inner"
    )

    return final_df


# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():
    print("🔹 Loading data...")
    ats_df, features_df = load_data()

    print("🔹 Computing similarity features...")
    similarity_df = compute_similarity_features(features_df)

    print("🔹 Merging datasets...")
    final_df = merge_all(ats_df, features_df, similarity_df)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Final dataset created!")
    print(f"Rows: {len(final_df)}")
    print(final_df.head())


if __name__ == "__main__":
    main()