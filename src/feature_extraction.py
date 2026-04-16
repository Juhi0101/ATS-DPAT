# =========================================
# ADVANCED FEATURE ENGINEERING (FINAL)
# =========================================

import os
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ------------------------------
# CONFIG
# ------------------------------

RESUME_FOLDER = "../data/resumes/"
SKILLS_FILE = "../data/skills.txt"
VERBS_FILE = "../data/verbs.txt"

OUTPUT_PATH = "../data/processed/features.csv"

os.makedirs("../data/processed", exist_ok=True)

# ------------------------------
# LOAD LISTS (NO HARDCODING)
# ------------------------------

def load_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

# Base lists
BASE_SKILLS = load_list(SKILLS_FILE)
ACTION_VERBS = load_list(VERBS_FILE)
STOPWORDS = set(ENGLISH_STOP_WORDS)


# ------------------------------
# TEXT CLEANING
# ------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


# ------------------------------
# DYNAMIC SKILL EXTRACTION
# ------------------------------

def extract_dynamic_skills(texts, top_n=50):
    word_counter = Counter()

    for text in texts:
        words = clean_text(text).split()
        word_counter.update(words)

    common_words = [w for w, _ in word_counter.most_common(top_n)]

    return common_words


# ------------------------------
# FEATURE FUNCTIONS
# ------------------------------

def basic_features(words):
    word_count = len(words)
    unique_words = len(set(words))

    return {
        "word_count": word_count,
        "unique_word_ratio": unique_words / word_count if word_count else 0,
        "avg_word_length": sum(len(w) for w in words) / word_count if word_count else 0
    }


def content_features(text, words):
    action_count = sum(1 for w in words if w in ACTION_VERBS)
    number_count = len(re.findall(r"\d+", text))

    return {
        "action_verb_density": action_count / len(words) if words else 0,
        "numeric_density": number_count / len(words) if words else 0,
        "bullet_points_estimate": text.count("•") + text.count("-")
    }


def skill_features(text, words, skills):
    matched = [skill for skill in skills if skill in text]

    return {
        "skills_count": len(matched),
        "skill_density": len(matched) / len(words) if words else 0,
        "matched_skills": ", ".join(matched)
    }


def quality_features(text, words):
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    uppercase_count = sum(1 for c in text if c.isupper())

    return {
        "stopword_ratio": stopword_count / len(words) if words else 0,
        "uppercase_ratio": uppercase_count / len(text) if text else 0
    }


# ------------------------------
# MAIN FEATURE EXTRACTION
# ------------------------------

def extract_features(text, skills):
    cleaned = clean_text(text)
    words = cleaned.split()

    features = {}

    features.update(basic_features(words))
    features.update(content_features(text, words))
    features.update(skill_features(cleaned, words, skills))
    features.update(quality_features(text, words))

    return features


# ------------------------------
# PROCESS ALL RESUMES
# ------------------------------

def process_resumes(folder):
    texts = []
    file_data = []

    # First pass: collect texts
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            texts.append(text)
            file_data.append((file, text))

    # Dynamic skill expansion
    dynamic_skills = extract_dynamic_skills(texts, top_n=50)
    FINAL_SKILLS = list(set(BASE_SKILLS) | set(dynamic_skills))

    print(f"✅ Loaded {len(BASE_SKILLS)} base skills")
    print(f"✅ Added {len(dynamic_skills)} dynamic skills")

    all_data = []

    for file, text in file_data:
        features = extract_features(text, FINAL_SKILLS)

        # Parse filename: R01_A.txt
        name = file.replace(".txt", "")
        resume_id, version = name.split("_")

        features["resume_id"] = resume_id
        features["version"] = version

        all_data.append(features)

    return pd.DataFrame(all_data)


# ------------------------------
# SAVE OUTPUT
# ------------------------------

def main():
    df = process_resumes(RESUME_FOLDER)
    print(df) ###
    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Feature extraction complete.")
    print(df.head())


if __name__ == "__main__":
    main()