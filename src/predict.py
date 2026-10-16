import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import CERTIFICATES, LEVELS, SOFT_SKILLS, TECH_SKILLS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "placement_model_v1.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

FEATURES = ["CGPA", "technical_skills", "soft_skills", "international_certificates",
            "extracurricular_activities", "club_membership", "real_world_projects",
            "past_work_experience", "problem_solving_skills", "communication_skills"]


def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def _validate(p):
    missing = [f for f in FEATURES if f not in p]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    if not 0.0 <= float(p["CGPA"]) <= 4.0:
        raise ValueError("CGPA must be between 0.0 and 4.0")
    if not 0 <= int(p["real_world_projects"]) <= 5:
        raise ValueError("Projects must be between 0 and 5 (range seen in training data)")
    if not 0 <= int(p["past_work_experience"]) <= 4:
        raise ValueError("Work experience must be between 0 and 4 (range seen in training data)")
    for col in ("problem_solving_skills", "communication_skills"):
        if p[col] not in LEVELS:
            raise ValueError(f"{col} must be one of {LEVELS}")
    for col, catalog in (("technical_skills", TECH_SKILLS), ("soft_skills", SOFT_SKILLS),
                         ("international_certificates", CERTIFICATES)):
        bad = [i for i in p[col] if i not in catalog]
        if bad:
            raise ValueError(f"Unknown items in {col}: {bad}")


def predict_placement(profile, model=None, meta=None):
    """profile: dict with the 10 FEATURES (list-type fields are Python lists)."""
    if model is None or meta is None:
        model, meta = load_model()
    _validate(profile)
    row = {k: profile[k] for k in FEATURES}
    for col in ("technical_skills", "soft_skills", "international_certificates"):
        row[col] = ", ".join(row[col]) if row[col] else "None"
    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[0, 1])
    placed = prob >= meta["threshold"]
    return {"placed": bool(placed),
            "label": "Likely to be Placed" if placed else "Lower Likelihood of Placement",
            "probability": round(prob, 4),
            "probability_raw": prob,
            "threshold": meta["threshold"],
            "note": "Model estimate based on the entered profile; not a guarantee of any outcome."}
