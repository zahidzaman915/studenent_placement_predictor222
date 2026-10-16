import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TECH_SKILLS = ["AWS", "Algorithms", "C++", "Cloud Computing", "Computer Vision", "Data Structures",
               "Data Visualization", "Deep Learning", "Docker", "Excel", "FastAPI", "Flask", "Git",
               "GitHub", "Java", "Machine Learning", "Matplotlib", "NLP", "NumPy", "OOP", "Pandas",
               "Power BI", "Python", "PyTorch", "SQL", "Scikit-learn", "Statistics", "TensorFlow"]
SOFT_SKILLS = ["Adaptability", "Collaboration", "Creativity", "Critical Thinking", "Decision Making",
               "Leadership", "Negotiation", "Presentation", "Teamwork", "Time Management"]
CERTIFICATES = ["AWS Certificate", "Cisco Certificate", "Coursera", "DigiSkills", "Google Certificate",
                "IBM Certificate", "Microsoft Certificate", "NAVTTC", "Udemy"]
LEVELS = ["Low", "Basic", "Good", "Very Good", "Excellent"]

LIST_COLS = {"technical_skills": (TECH_SKILLS, "tech", "tech_skill_count"),
             "soft_skills": (SOFT_SKILLS, "soft", "soft_skill_count"),
             "international_certificates": (CERTIFICATES, "cert", "cert_count")}

def _slug(t):
    return t.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")

def _items(s):
    return set() if pd.isna(s) or s in ("", "None") else {t.strip() for t in s.split(",")}

BINARY_COLS = [f"{p}_{_slug(i)}" for cat, p, _ in LIST_COLS.values() for i in cat]
NUMERIC_COLS = ["age", "CGPA", "real_world_projects", "past_work_experience",
                "tech_skill_count", "soft_skill_count", "cert_count"]
ORDINAL_COLS = ["problem_solving_skills", "communication_skills"]
NOMINAL_COLS = ["gender", "extracurricular_activities", "club_membership"]

def engineer_features(X):
    """Stateless: turns list-like text columns into counts + 0/1 indicators."""
    X = X.copy()
    for col, (catalog, prefix, count_name) in LIST_COLS.items():
        sets = X[col].apply(_items)
        X[count_name] = sets.apply(len)
        for item in catalog:
            X[f"{prefix}_{_slug(item)}"] = sets.apply(lambda s, i=item: int(i in s))
    return X.drop(columns=list(LIST_COLS))

def build_preprocessor(scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    ord_steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                 ("encoder", OrdinalEncoder(categories=[LEVELS, LEVELS],
                                            handle_unknown="use_encoded_value", unknown_value=-1))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
        ord_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_COLS),
        ("ord", Pipeline(ord_steps), ORDINAL_COLS),
        ("nom", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), NOMINAL_COLS),
        ("bin", "passthrough", BINARY_COLS)])


def build_subset_preprocessor(scale=False, exclude=()):
    """Same as build_preprocessor but drops the listed columns (raw or engineered)."""
    prep = build_preprocessor(scale=scale)
    new = []
    for name, trans, cols in prep.transformers:
        cols = [c for c in cols if c not in exclude]
        if name == "ord" and cols:
            trans.named_steps["encoder"].set_params(categories=[LEVELS] * len(cols))
        new.append((name, trans, cols))
    prep.set_params(transformers=new)
    return prep
