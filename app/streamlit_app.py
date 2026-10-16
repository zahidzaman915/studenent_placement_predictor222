import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import CERTIFICATES, LEVELS, SOFT_SKILLS, TECH_SKILLS  # noqa: E402
from src.predict import load_model, predict_placement  # noqa: E402
from src.recommend import improvement_levers, profile_strengths  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("placement_app")

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

CLUBS = ["None", "DSCS - Member", "DSCS - Active Member", "DSCS - Event Organizer", "DSCS - Executive",
         "University Tech Club - Member", "University Tech Club - Executive"]
ACTIVITIES = ["None", "Community Service", "Debate", "Hackathon", "Research Activity", "Sports",
              "Tech Event Organizer", "Volunteer Work", "Workshop Organizer"]

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg, #12294d 0%, #1F4E8C 60%, #2f7bd1 100%);
       padding: 2rem 2.2rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 2.1rem;}
.hero p {color: #dce8f7; margin: 0; font-size: 1.02rem;}
.sec {font-weight: 700; color: #12294d; font-size: 1.05rem; margin-bottom: .4rem;}
.result {padding: 1.4rem 1.6rem; border-radius: 14px; margin: .8rem 0 1rem 0; color: #1B2430;}
.result h2 {margin: 0 0 .5rem 0;}
.result.good {background: #E8F5E9; border-left: 8px solid #2E7D32;}
.result.low {background: #EAF2FB; border-left: 8px solid #1F4E8C;}
.tip {background: #fff; border: 1px solid #dfe6f1; border-radius: 12px; padding: .8rem 1rem;
      margin-bottom: .6rem; color: #1B2430;}
.tip b {color: #12294d;}
.gain {color: #2E7D32; font-weight: 600; font-size: .9rem;}
.small {color: #5b6676; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)


def fmt_pct(p):
    return "<1%" if p < 0.01 else ">99%" if p > 0.99 else f"{p:.0%}"


@st.cache_resource
def get_model():
    return load_model()


try:
    model, meta = get_model()
except Exception as exc:  # pragma: no cover
    st.error("The prediction model could not be loaded. Please contact the project owner.")
    log.exception("Model load failed: %s", exc)
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🎓 Student Placement Predictor</h1>
  <p>Estimate placement likelihood from academic, technical and professional profile factors,
  and get personalised guidance on how to strengthen it.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About the Project")
    st.write("A machine-learning system that predicts whether a student is likely to be placed, "
             "trained on 5,000 synthetic student records.")
    tm = meta["test_metrics"]
    st.markdown("**Model:** Logistic Regression  \n"
                f"**Recall (placed students):** {tm['recall']:.0%}  \n"
                f"**Precision:** {tm['precision']:.0%}  \n"
                f"**ROC-AUC:** {tm['roc_auc']:.2f}")
    st.markdown("**Privacy:** no name, CNIC, gender or age is collected or used.")
    st.caption("Results are statistical estimates for guidance only. They are not a guarantee "
               "of any hiring or placement outcome.")

with st.form("profile"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="sec">1 · Student Academic Profile</div>', unsafe_allow_html=True)
            cgpa = st.slider("CGPA (out of 4.0)", 0.0, 4.0, 3.0, 0.01, key="cgpa")
        with st.container(border=True):
            st.markdown('<div class="sec">2 · Technical Skills</div>', unsafe_allow_html=True)
            tech = st.multiselect("Select your technical skills", TECH_SKILLS, key="tech")
        with st.container(border=True):
            st.markdown('<div class="sec">3 · Soft Skills</div>', unsafe_allow_html=True)
            soft = st.multiselect("Select your soft skills", SOFT_SKILLS, key="soft")
        with st.container(border=True):
            st.markdown('<div class="sec">4 · Certifications</div>', unsafe_allow_html=True)
            certs = st.multiselect("Certificates earned", CERTIFICATES, key="certs")
    with c2:
        with st.container(border=True):
            st.markdown('<div class="sec">5 · Projects</div>', unsafe_allow_html=True)
            projects = st.slider("Real-world projects completed", 0, 5, 1, key="projects")
        with st.container(border=True):
            st.markdown('<div class="sec">6 · Work Experience</div>', unsafe_allow_html=True)
            exp = st.slider("Internships / jobs / freelance experiences", 0, 4, 0, key="exp")
        with st.container(border=True):
            st.markdown('<div class="sec">7 · Problem-Solving Skills</div>', unsafe_allow_html=True)
            ps = st.select_slider("Your level", LEVELS, value="Good", key="ps")
        with st.container(border=True):
            st.markdown('<div class="sec">8 · Communication Skills</div>', unsafe_allow_html=True)
            comm = st.select_slider("Your level", LEVELS, value="Good", key="comm")
        with st.container(border=True):
            st.markdown('<div class="sec">Activities and Clubs</div>', unsafe_allow_html=True)
            club = st.selectbox("Club membership", CLUBS, key="club")
            act = st.selectbox("Extracurricular activity", ACTIVITIES, key="act")
    submitted = st.form_submit_button("Predict Placement Likelihood", type="primary",
                                      use_container_width=True)

if submitted:
    profile = {"CGPA": cgpa, "technical_skills": tech, "soft_skills": soft,
               "international_certificates": certs, "extracurricular_activities": act,
               "club_membership": club, "real_world_projects": projects,
               "past_work_experience": exp, "problem_solving_skills": ps,
               "communication_skills": comm}
    try:
        res = predict_placement(profile, model, meta)
    except ValueError as err:
        st.error(f"Please check your inputs: {err}")
        st.stop()
    log.info("prediction made: placed=%s prob=%.3f", res["placed"], res["probability"])
    prob = res["probability"]

    st.markdown("---")
    st.subheader("Prediction Result")
    if res["placed"]:
        st.markdown("""
<div class="result good"><h2>🎉 Congratulations!</h2>
<p>Based on your academic performance, technical skills, projects, experience and other relevant
factors, the model predicts a <b>strong likelihood of placement</b> for your profile.</p></div>
""", unsafe_allow_html=True)
    else:
        near = (meta["threshold"] - prob) <= 0.10
        extra = " You are close to the positive range, so a few targeted improvements could make a difference." if near else ""
        st.markdown(f"""
<div class="result low"><h2>Your Placement Outlook</h2>
<p>Thank you for completing the assessment. Based on the current profile, the model predicts a
<b>lower likelihood of placement at this stage</b>. This is a snapshot of today's profile, not a
verdict on your potential.{extra}</p></div>
""", unsafe_allow_html=True)

    m1, m2 = st.columns([1, 2])
    m1.metric("Estimated placement likelihood", fmt_pct(prob))
    with m2:
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"The model classifies a profile as 'likely placed' at or above "
                   f"{meta['threshold']:.0%}. This is an estimate, not a guarantee.")

    if res["placed"]:
        strengths = profile_strengths(profile)
        if strengths:
            st.markdown("#### What is working in your favour")
            for s in strengths:
                st.markdown(f"- {s}")
        levers = improvement_levers(profile, model, meta, top_n=2)
        if levers:
            st.markdown("#### To stay competitive")
            for lv in levers:
                st.markdown(f'<div class="tip"><b>{lv["title"]}</b><br>{lv["detail"]}</div>',
                            unsafe_allow_html=True)
    else:
        levers = improvement_levers(profile, model, meta, top_n=4)
        st.markdown("#### Recommended next steps for your profile")
        if levers:
            for lv in levers:
                st.markdown(
                    f'<div class="tip"><b>{lv["title"]}</b> '
                    f'<span class="gain">({lv["impact"]})</span>'
                    f'<br>{lv["detail"]}</div>', unsafe_allow_html=True)
        else:
            st.info("Your profile is already strong across the factors the model considers.")

    with st.expander("How to read this result"):
        st.markdown(
            "- The percentage is the model's estimated likelihood, based on patterns in 5,000 "
            "synthetic student records.\n"
            "- Recommendations come from re-scoring your profile with one improvement at a time. "
            "The impact label reflects how much the model's estimate would move. It shows association, not proven cause and effect.\n"
            "- Real hiring depends on many factors this model cannot see.")
