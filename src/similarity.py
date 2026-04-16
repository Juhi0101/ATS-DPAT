# =========================================
# ADVANCED ATS MATCHING ENGINE (IMPROVED)
# =========================================

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ------------------------------
# LOAD LISTS
# ------------------------------

def load_list(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Required skill list not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_FILE = "_data/skills.txt"
SKILLS = load_list(SKILLS_FILE)

# STOPWORDS_FILE = BASE_DIR / "data" / "stopwords.txt"
# STOPWORDS = set(load_list(STOPWORDS_FILE))


DOMAIN_STOPWORDS = {
    "experience", "skills", "knowledge", "ability",
    "team", "project", "role", "responsible",
    "requirements", "candidate", "job",
    "work", "worked", "working", "looking", "skilled",
    "tools", "technologies", "proficient", "familiar",
    "strong", "excellent", "good", "ability", "responsible",
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS

# ------------------------------
# CLEAN TEXT
# ------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# TF-IDF SIMILARITY (IMPROVED)
# ------------------------------

def compute_similarity(resume, job):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # BIG upgrade: captures phrases
        max_features=1000
    )

    corpus = [resume, job]
    tfidf = vectorizer.fit_transform(corpus)

    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return score, vectorizer, tfidf


# ------------------------------
# KEYWORD MATCHING
# ------------------------------

def keyword_analysis(vectorizer, tfidf_matrix, stopwords, skills):
    features = np.array(vectorizer.get_feature_names_out())

    resume_vec = tfidf_matrix[0].toarray().flatten()
    job_vec = tfidf_matrix[1].toarray().flatten()

    resume_keywords = set(features[resume_vec > 0])
    job_keywords = set(features[job_vec > 0])

    # ------------------------------
    # CLEANING STEP
    # ------------------------------

    def clean_keywords(keywords):
        cleaned = set()

        for word in keywords:
            if word in stopwords:
                continue

            # ❌ REMOVE random phrases like "python flask"
            if " " in word:
                # keep only if it's a known skill
                if word in skills:
                    cleaned.add(word)
            else:
                cleaned.add(word)

        return cleaned

    resume_keywords = clean_keywords(resume_keywords)
    job_keywords = clean_keywords(job_keywords)

    # ------------------------------
    # MATCHING
    # ------------------------------

    matched = resume_keywords & job_keywords
    missing = job_keywords - resume_keywords

    # ------------------------------
    # SCORING (skill priority)
    # ------------------------------

    skill_keywords = {w for w in job_keywords if w in skills}
    skill_matches = matched & skills

    if len(skill_keywords) == 0:
        skill_part = 0
    else:
        skill_part = len(skill_matches) / len(skill_keywords)

    keyword_part = len(matched) / len(job_keywords) if job_keywords else 0

    coverage = (0.7 * skill_part) + (0.3 * keyword_part)

    return matched, missing, coverage


# ------------------------------
# SKILL MATCHING (IMPROVED)
# ------------------------------

def skill_matching(resume_text, job_text):
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)

    resume_skills = set()
    job_skills = set()

    for skill in SKILLS:
        if skill in resume_text:
            resume_skills.add(skill)
        if skill in job_text:
            job_skills.add(skill)

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills

    score = len(matched) / len(job_skills) if job_skills else 0

    return matched, missing, score, resume_skills


# ------------------------------
# FINAL SCORING
# ------------------------------

def compute_final_score(similarity, skill_score, keyword_score,
                        matched_skills, resume_skills):

    total_resume_skills = len(resume_skills)
    total_matched = len(matched_skills)

    # ------------------------------
    # PRECISION
    # ------------------------------

    total_resume_skills = len(resume_skills)
    total_matched = len(matched_skills)

    if total_resume_skills == 0:
        precision_factor = 0.9  # neutral fallback
    else:
        precision_ratio = total_matched / total_resume_skills
        extra_skills = total_resume_skills - total_matched

        # ------------------------------
        # SOFT PENALTY LOGIC
        # ------------------------------

        if extra_skills <= 2:
            #  Small extra skills → NO penalty
            precision_factor = 1.0

        elif extra_skills <= 5:
            #  Moderate extras → slight penalty
            precision_factor = 0.95 * precision_ratio + 0.05

        else:
            #  Too many irrelevant skills → stronger penalty
            precision_factor = 0.85 * precision_ratio
    # ------------------------------
    # FINAL SCORE
    # ------------------------------
    final = (
        0.25 * similarity +
        0.55 * skill_score +
        0.10 * keyword_score +
        0.10 * precision_factor
    )

    return round(final * 100, 2)


def compute_feature_importance(similarity, skill_score, keyword_score, matched_skills, missing_skills):
    similarity_pct = round(similarity * 100, 1)
    skill_pct = round(skill_score * 100, 1)
    keyword_pct = round(keyword_score * 100, 1)

    factors = [
        {
            "result": "Semantic similarity",
            "reason": f"Your resume and job description share {similarity_pct}% of important language and concepts."
        },
        {
            "result": "Skill coverage",
            "reason": f"You matched {skill_pct}% of the job skills ({len(matched_skills)} skills)."
        },
    ]

    if missing_skills:
        missing_snapshot = ', '.join(list(missing_skills)[:4])
        factors.append({
            "result": "Missing skills",
            "reason": f"Key skills not found in your resume: {missing_snapshot}{'...' if len(missing_skills) > 4 else ''}."
        })
    else:
        factors.append({
            "result": "No missing skills",
            "reason": "Your resume contains all required skills for this job description."
        })

    keyword_factor = {
        "result": "Keyword coverage",
        "reason": f"Keyword coverage is {'strong' if keyword_score >= 0.5 else 'low'} at {keyword_pct}%.",
    }
    factors.append(keyword_factor)

    return factors[:3]


# ------------------------------
# EXPLANATION GENERATOR
# ------------------------------

def generate_explanation(similarity, skill_score, keyword_score):
    explanation = []

    if similarity > 0.6:
        explanation.append("Strong alignment with job description")
    elif similarity > 0.3:
        explanation.append("Moderate relevance to job description")
    else:
        explanation.append("Low relevance to job description")

    if skill_score < 0.5:
        explanation.append("Missing important skills")
    else:
        explanation.append("Good skill match")

    if keyword_score < 0.5:
        explanation.append("Insufficient keyword coverage")

    return "; ".join(explanation)


# ------------------------------
# MAIN FUNCTION
# ------------------------------

def ats_match(resume_text, job_text):
    resume = clean_text(resume_text)
    job = clean_text(job_text)

    similarity, vectorizer, tfidf_matrix = compute_similarity(resume, job)

    matched_kw, missing_kw, keyword_score = keyword_analysis(
        vectorizer,
        tfidf_matrix,
        STOPWORDS,
        set(SKILLS)
    )

    matched_skills, missing_skills, skill_score, resume_skills = skill_matching(resume, job)

    final_score = compute_final_score(
        similarity,
        skill_score,
        keyword_score,
        matched_skills,
        resume_skills
    )

    explanation = generate_explanation(similarity, skill_score, keyword_score)
    feature_importance = compute_feature_importance(
        similarity,
        skill_score,
        keyword_score,
        matched_skills,
        missing_skills,
    )

    return {
        "similarity_score": round(similarity * 100, 2),
        "final_score": final_score,
        "skill_score": round(skill_score * 100, 2),
        "keyword_score": round(keyword_score * 100, 2),
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills),
        "matched_keywords": list(matched_kw)[:20],
        "missing_keywords": list(missing_kw)[:20],
        "feature_importance": feature_importance,
        "explanation": explanation
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