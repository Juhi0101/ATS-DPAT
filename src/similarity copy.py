# =========================================
# ADVANCED SIMILARITY ENGINE (ATS CORE)
# =========================================

import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_FILE = BASE_DIR / "data" / "skills.txt"


# ------------------------------
# LOAD SKILLS
# ------------------------------

def load_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

BASE_SKILLS = load_list(SKILLS_FILE)


# ------------------------------
# TEXT CLEANING
# ------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


# ------------------------------
# TF-IDF SIMILARITY
# ------------------------------

def compute_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english")

    corpus = [resume_text, job_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity, vectorizer, tfidf_matrix


# ------------------------------
# KEYWORD MATCHING
# ------------------------------

def extract_keywords(vectorizer, tfidf_matrix):
    feature_names = np.array(vectorizer.get_feature_names_out())

    resume_vec = tfidf_matrix[0].toarray().flatten()
    job_vec = tfidf_matrix[1].toarray().flatten()

    resume_keywords = set(feature_names[resume_vec > 0])
    job_keywords = set(feature_names[job_vec > 0])

    matched = resume_keywords & job_keywords
    missing = job_keywords - resume_keywords

    return {
        "matched_keywords": list(matched),
        "missing_keywords": list(missing)
    }


# ------------------------------
# SKILL MATCHING
# ------------------------------

def skill_matching(resume_text, job_text, skills):
    resume_skills = {skill for skill in skills if skill in resume_text}
    job_skills = {skill for skill in skills if skill in job_text}

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills

    return {
        "matched_skills": list(matched),
        "missing_skills": list(missing)
    }


# ------------------------------
# FINAL ATS SCORE
# ------------------------------

def compute_final_score(similarity, matched_skills, total_job_skills):
    if total_job_skills == 0:
        skill_score = 0
    else:
        skill_score = len(matched_skills) / total_job_skills

    # Weighted score
    final_score = (0.7 * similarity) + (0.3 * skill_score)

    return round(final_score * 100, 2)


def compute_feature_importance(similarity, matched_skills, missing_skills, total_job_skills):
    if total_job_skills == 0:
        return [
            "No job skills were detected, so feature importance cannot be computed.",
        ]

    similarity_pct = round(similarity * 100, 2)
    skill_ratio = len(matched_skills) / total_job_skills
    matched_ratio_pct = round(skill_ratio * 100, 2)

    importance = [
        (
            0.7 * similarity,
            f"Strong semantic similarity: your resume and the job description share {similarity_pct}% of key language and terminology."
        ),
        (
            0.3 * skill_ratio,
            f"Skill coverage: your resume matches {len(matched_skills)} of {total_job_skills} required job skills ({matched_ratio_pct}%)."
        ),
    ]

    if missing_skills:
        missing_slice = ", ".join(missing_skills[:5])
        ellipsis = "..." if len(missing_skills) > 5 else ""
        importance.append(
            (
                0.3 * (1 - skill_ratio),
                f"Missing skills impact: key skills not found in your resume include {missing_slice}{ellipsis}."
            )
        )
    else:
        importance.append(
            (
                0.3,
                "No required skills are missing, maximizing your skill-match component."
            )
        )

    importance.sort(key=lambda item: item[0], reverse=True)
    return [description for _, description in importance[:3]]


# ------------------------------
# MAIN PIPELINE
# ------------------------------

def ats_match(resume_text, job_text):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    similarity, vectorizer, tfidf_matrix = compute_similarity(resume_clean, job_clean)

    keyword_data = extract_keywords(vectorizer, tfidf_matrix)
    skill_data = skill_matching(resume_clean, job_clean, BASE_SKILLS)

    final_score = compute_final_score(
        similarity,
        skill_data["matched_skills"],
        len(skill_data["matched_skills"]) + len(skill_data["missing_skills"])
    )

    return {
        "similarity_score": float(round(similarity * 100, 2)),
        "final_score": float(final_score),
        "matched_keywords": keyword_data["matched_keywords"],
        "missing_keywords": keyword_data["missing_keywords"],
        "matched_skills": skill_data["matched_skills"],
        "missing_skills": skill_data["missing_skills"],
        "feature_importance": compute_feature_importance(
            similarity,
            skill_data["matched_skills"],
            skill_data["missing_skills"],
            len(skill_data["matched_skills"]) + len(skill_data["missing_skills"])
        )
    }


# ------------------------------
# TEST EXAMPLE
# ------------------------------

if __name__ == "__main__":
    resume_text = """Experienced Python developer with knowledge of machine learning and data analysis."""
    job_text = """Looking for a Python developer skilled in data analysis, machine learning, Business Intelligence Tools, and deep learning."""

    result = ats_match(resume_text, job_text)

    for key, value in result.items():
        print(f"{key}: {value}")