# =========================================
# SKILL EXTRACTION FROM STRUCTURED DATASET
# =========================================

import pandas as pd
import re

# ------------------------------
# CONFIG
# ------------------------------

DATA_PATH = "../data/technical_skills.csv"   # your new dataset
OUTPUT_PATH = "../data/skills.txt"


# ------------------------------
# CLEAN SKILL NAME
# ------------------------------

def clean_skill(skill):
    skill = str(skill).lower().strip()

    # Remove extra symbols but KEEP + and # (for C++, C#)
    skill = re.sub(r"[^a-z0-9+#\s]", "", skill)

    skill = re.sub(r"\s+", " ", skill)

    return skill


# ------------------------------
# LOAD AND EXTRACT
# ------------------------------

def extract_skills(path):
    df = pd.read_csv(path)

    if "Skill Name" not in df.columns:
        raise ValueError("❌ Column 'Skill Name' not found")

    skills = df["Skill Name"].dropna().tolist()

    cleaned_skills = set()

    for skill in skills:
        cleaned = clean_skill(skill)

        if len(cleaned) > 1:
            cleaned_skills.add(cleaned)

    return sorted(cleaned_skills)


# ------------------------------
# SAVE
# ------------------------------

def save_skills(skills, path):
    with open(path, "w", encoding="utf-8") as f:
        for skill in skills:
            f.write(skill + "\n")

    print(f"✅ Saved {len(skills)} skills to {path}")


# ------------------------------
# MAIN
# ------------------------------

def main():
    print("🔹 Loading structured skill dataset...")

    skills = extract_skills(DATA_PATH)

    print(f"🔹 Extracted {len(skills)} skills")

    save_skills(skills, OUTPUT_PATH)


if __name__ == "__main__":
    main()