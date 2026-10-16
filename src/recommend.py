import math

from src.features import LEVELS
from src.predict import predict_placement

TECH_PRIORITY = ["Python", "SQL", "Machine Learning", "Git", "Pandas", "Data Visualization",
                 "Statistics", "Docker", "Cloud Computing"]
CERT_PRIORITY = ["Google Certificate", "Coursera", "Microsoft Certificate", "IBM Certificate",
                 "AWS Certificate", "Udemy"]
SOFT_PRIORITY = ["Leadership", "Presentation", "Critical Thinking", "Teamwork", "Time Management"]


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _next_level(level):
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


def _missing(owned, priority, k):
    return [x for x in priority if x not in owned][:k]


def improvement_levers(profile, model, meta, top_n=4):
    """What-if analysis: apply one realistic improvement at a time, re-score it with the
    model, and return the actions with the largest estimated gain for THIS profile."""
    base = _logit(predict_placement(profile, model, meta)["probability_raw"])
    found = []

    def add(title, detail, changes):
        gain = _logit(predict_placement({**profile, **changes}, model, meta)["probability_raw"]) - base
        if gain > 0.02:
            impact = "High impact" if gain >= 0.5 else "Medium impact" if gain >= 0.25 else "Helpful"
            found.append({"title": title, "detail": detail, "gain": gain, "impact": impact})

    cg = float(profile["CGPA"])
    if cg < 3.8:
        tgt = round(min(4.0, cg + 0.3), 2)
        add("Raise your CGPA", f"Work toward about {tgt:.2f} through consistent study and by "
            "strengthening your weaker courses.", {"CGPA": tgt})

    n = int(profile["real_world_projects"])
    if n < 5:
        add("Build more real-world projects", f"You listed {n}. Add a practical project (data analysis, "
            "machine learning or a web app) and publish it on GitHub.", {"real_world_projects": n + 1})

    tech = list(profile["technical_skills"])
    if len(tech) < 8:
        new = _missing(tech, TECH_PRIORITY, min(2, 8 - len(tech)))
        if new:
            add("Expand your technical skills", f"You listed {len(tech)}. Consider learning "
                f"{' and '.join(new)}.", {"technical_skills": tech + new})

    e = int(profile["past_work_experience"])
    if e < 4:
        add("Gain practical work experience", f"You listed {e}. Look for internships, freelance work, "
            "part-time roles or open-source contributions.", {"past_work_experience": e + 1})

    certs = list(profile["international_certificates"])
    if len(certs) < 3:
        new = _missing(certs, CERT_PRIORITY, 1)
        if new:
            add("Earn a recognised certification", f"You listed {len(certs)}. A certificate such as "
                f"{new[0]} would strengthen your profile.", {"international_certificates": certs + new})

    soft = list(profile["soft_skills"])
    if len(soft) < 6:
        new = _missing(soft, SOFT_PRIORITY, 1)
        if new:
            add("Develop your soft skills", f"You listed {len(soft)}. Build {new[0].lower()} through "
                "team projects and student societies.", {"soft_skills": soft + new})

    if profile["communication_skills"] != LEVELS[-1]:
        add("Strengthen communication", "Practise presentations, join debate or speaking sessions, "
            "and rehearse interview answers.",
            {"communication_skills": _next_level(profile["communication_skills"])})

    if profile["problem_solving_skills"] != LEVELS[-1]:
        add("Sharpen problem-solving", "Solve coding and logic problems regularly "
            "(algorithms, data structures, case studies).",
            {"problem_solving_skills": _next_level(profile["problem_solving_skills"])})

    if profile["club_membership"] == "None":
        add("Join a technical club", "Active club membership shows initiative and builds your network.",
            {"club_membership": "DSCS - Member"})

    if profile["extracurricular_activities"] == "None":
        add("Take part in activities", "Hackathons, workshops or volunteering add visible experience.",
            {"extracurricular_activities": "Hackathon"})

    return sorted(found, key=lambda d: d["gain"], reverse=True)[:top_n]


def profile_strengths(profile):
    s = []
    if float(profile["CGPA"]) >= 3.3:
        s.append("Strong academic record (CGPA)")
    if len(profile["technical_skills"]) >= 6:
        s.append("Broad technical skill set")
    if int(profile["real_world_projects"]) >= 3:
        s.append("Solid portfolio of real-world projects")
    if int(profile["past_work_experience"]) >= 2:
        s.append("Relevant work experience")
    if len(profile["international_certificates"]) >= 2:
        s.append("Multiple professional certifications")
    if profile["problem_solving_skills"] in LEVELS[-2:]:
        s.append("Strong problem-solving ability")
    return s
