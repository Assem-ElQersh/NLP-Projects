from __future__ import annotations

import re
from typing import List, Tuple, Dict

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def extract_skills(text: str, skill_list: List[str]) -> List[str]:
    """
    Very simple skill extractor via case-insensitive substring matching.
    """
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        skill_norm = skill.strip().lower()
        if len(skill_norm) == 0:
            continue
        if skill_norm in text_lower:
            found.append(skill)
    return list(dict.fromkeys(found))


def build_feature_text(job_desc: str, resume: str) -> Tuple[str, str]:
    return basic_clean(job_desc), basic_clean(resume)


DEFAULT_SKILLS = [
    "python", "java", "c++", "sql", "scala", "pyspark", "aws", "azure", "gcp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "computer vision", "time series", "statistics", "machine learning",
    "deep learning", "docker", "kubernetes", "airflow", "spark", "hadoop",
    "data engineering", "data analysis", "etl", "mlops",
]


def add_skill_overlap_features(df: pd.DataFrame, skills: List[str] = None) -> pd.DataFrame:
    if skills is None:
        skills = DEFAULT_SKILLS
    out = df.copy()
    jd_skills = []
    cv_skills = []
    overlaps = []
    for jd, cv in zip(out["job_description"], out["resume"]):
        jd_clean, cv_clean = build_feature_text(jd, cv)
        jd_s = set(extract_skills(jd_clean, skills))
        cv_s = set(extract_skills(cv_clean, skills))
        overlap = len(jd_s & cv_s)
        jd_skills.append(";".join(sorted(jd_s)))
        cv_skills.append(";".join(sorted(cv_s)))
        overlaps.append(overlap)
    out["jd_skills"] = jd_skills
    out["resume_skills"] = cv_skills
    out["skill_overlap"] = overlaps
    return out


