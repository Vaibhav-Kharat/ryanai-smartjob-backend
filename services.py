import json
import re
import google.generativeai as genai
import spacy
import jwt
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import joinedload, load_only
from sqlalchemy.orm import Session, joinedload, Session
from models import Job, CandidateProfile, EmployerProfile, Category, User
from utils import extract_job_keywords, extract_text_from_resume, extract_resume_keywords, generate_match_reason
from config import settings
from datetime import datetime
from logger import gemini_logger  # Import the logger
# This is the correct class name
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel

nlp = spacy.load("en_core_web_sm")
genai.configure(api_key=settings.GOOGLE_API_KEY)
# Load env variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
security = HTTPBearer()


# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
def verify_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        return None  # No token provided → fallback to body

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def process_job(job_id: str, db: Session):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return None

    extracted = extract_job_keywords(job.description)

    # ✅ Store in keywords column (JSONB)
    job.keywords = extracted
    db.commit()
    db.refresh(job)

    return job
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


def verify_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        return None  # No token → fallback to body

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": True}  # validate expiry
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None   # expired → allow fallback
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def process_candidate(candidate_id: int, db: Session):
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == candidate_id).first()
    if not candidate:
        return None

    # 1. Extract text from resume
    resume_text = extract_text_from_resume(candidate.resumeUrl)
    

    # 2. Extract skills/experience using Gemini
    extracted = extract_resume_keywords(resume_text)

    # 3. Save in DB
    candidate.keywords = extracted
    db.commit()
    db.refresh(candidate)

    return candidate

# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


def normalize_skills(skills):
    if not skills:
        return set()
    if isinstance(skills, str):  # candidate case: comma-separated string
        return set([s.strip() for s in skills.split(",") if s.strip()])
    if isinstance(skills, list):  # job case
        return set([s.strip() for s in skills if isinstance(s, str) and s.strip()])
    return set()
# -------------------------------------------------------------------------------------------------#


def normalize_experience(exp):
    if not exp:
        return None
    if isinstance(exp, str):  # e.g. "2-4"
        return exp.strip()
    return str(exp).strip()
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY,
                             algorithms=[ALGORITHM])
        user_id = payload.get("candidate_id")  # Assume your JWT has 'user_id'
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def recommend_jobs(candidate_id: int, db: Session):
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == candidate_id
    ).first()
    if not candidate or not candidate.keywords:
        return []

    # Parse candidate keywords if stored as string
    candidate_keywords = candidate.keywords
    if isinstance(candidate_keywords, str):
        try:
            candidate_keywords = json.loads(candidate_keywords)
        except json.JSONDecodeError:
            candidate_keywords = {}

    # Parse candidate experience if it's a string or None
    candidate_exp_dict = candidate_keywords.get("experience")
    if candidate_exp_dict is None:
        candidate_exp_dict = {}
    elif isinstance(candidate_exp_dict, str):
        try:
            candidate_exp_dict = json.loads(candidate_exp_dict)
        except json.JSONDecodeError:
            candidate_exp_dict = {}

    candidate_skills = set(candidate_keywords.get("skills", []))
    candidate_exp = candidate_exp_dict.get("years") or 0

    jobs = db.query(Job).all()
    job_matches = []

    for job in jobs:
        # Parse job keywords if stored as string
        job_keywords = job.keywords or {}
        if isinstance(job_keywords, str):
            try:
                job_keywords = json.loads(job_keywords)
            except json.JSONDecodeError:
                job_keywords = {}  # Fallback to empty dict if parsing fails

        # Parse job experience if it's a string or None
        job_exp_dict = job_keywords.get("experience")
        if job_exp_dict is None:
            job_exp_dict = {}
        elif isinstance(job_exp_dict, str):
            try:
                job_exp_dict = json.loads(job_exp_dict)
            except json.JSONDecodeError:
                job_exp_dict = {}

        job_skills = set(job_keywords.get("skills", []))

        # Safely get experience values with defaults
        job_min_exp = 0
        job_max_exp = 0

        if isinstance(job_exp_dict, dict):
            job_min_exp = job_exp_dict.get("min_experience") or 0
            job_max_exp = job_exp_dict.get("max_experience") or 0

        # --- Skills match %
        skill_match_pct = (
            (len(candidate_skills.intersection(job_skills)) / len(job_skills)) * 100
            if job_skills else 0
        )

        # --- Experience match %
        if job_max_exp == 0:
            experience_match_pct = 0
        else:
            if job_min_exp <= candidate_exp <= job_max_exp:
                experience_match_pct = 100
            else:
                diff = min(abs(candidate_exp - job_min_exp),
                           abs(candidate_exp - job_max_exp))
                experience_match_pct = max(0, 100 - diff * 20)

        # --- Aggregate %
        aggregate_pct = (skill_match_pct + experience_match_pct) / 2
        reason = generate_match_reason(
            candidate_exp, candidate_skills, job_min_exp, job_max_exp, job_skills
        )
        job_matches.append({
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salaryMin": job.salaryMin,
            "salaryMax": job.salaryMax,
            "employer": {"companyName": job.employer.companyName if job.employer else None},
            "category": {"name": job.category.name if job.category else None},
            "skill_match_percentage": round(skill_match_pct),
            "experience_match_percentage": round(experience_match_pct),
            "aggregate_match_percentage": round(aggregate_pct),
            "match_reason": reason   # ✅ New AI-generated explanation
        })

    # Sort by aggregate %
    job_matches.sort(
        key=lambda x: x["aggregate_match_percentage"], reverse=True)
    # Return top 5
    return job_matches[:5]
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


def get_recommended_candidates_for_job(job_id: str, db: Session):
    """
    Recommend candidates for a given job based on job_id.
    """

    # Get the job
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return {"error": "Job not found"}

    job_keywords = job.keywords or {}
    if isinstance(job_keywords, str):
        import json
        try:
            job_keywords = json.loads(job_keywords)
        except json.JSONDecodeError:
            job_keywords = {}

    job_skills = set(job_keywords.get("skills", []))
    job_exp_dict = job_keywords.get("experience") or {}
    job_min_exp = job_exp_dict.get("min_experience") or 0
    job_max_exp = job_exp_dict.get("max_experience") or 0

    # Fetch all candidates
    candidates = (
        db.query(CandidateProfile, User, Category)
        .join(User, CandidateProfile.userId == User.id)
        .join(Category, CandidateProfile.categoryId == Category.id)
        .all()
    )

    recommended_candidates = []
    for candidate, user, category in candidates:
        candidate_skills = set(candidate.keywords.get(
            "skills", [])) if candidate.keywords else set()
        candidate_exp = candidate.totalExperience or 0

        # Skill match %
        skill_match_pct = (len(candidate_skills & job_skills) /
                           len(job_skills) * 100) if job_skills else 0

        # Experience match %
        if job_max_exp == 0:
            experience_match_pct = 0
        else:
            if job_min_exp <= candidate_exp <= job_max_exp:
                experience_match_pct = 100
            else:
                diff = min(abs(candidate_exp - job_min_exp),
                           abs(candidate_exp - job_max_exp))
                experience_match_pct = max(0, 100 - diff * 20)

        aggregate_pct = (skill_match_pct + experience_match_pct) / 2

        recommended_candidates.append({
            "id": candidate.id,
            "fullName": user.fullName,
            "image": user.image,
            "jobCategory": category.name if category else None,
            "currentLocation": candidate.currentLocation,
            "totalExperience": candidate_exp,
            "nationality": candidate.nationality,
            "resumeUrl": candidate.resumeUrl,
            "isBookmarked": False,
            "match_score": round(aggregate_pct)
        })

    # Sort by match score descending
    recommended_candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return recommended_candidates[:5]  # top 5 candidates

# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


def resume_parsing(resume_text: str):
    """
    Call Gemini to extract resume details in JSON format.
    """
    prompt = f"""
Extract resume details from the text below and return JSON in this format:

{{
  "personalDetails": {{
    "fullName": "<extract full name or N/A>",
    "phone": "<extract phone or N/A>",
    "currentLocation": "<extract current location or N/A>",
    "nationality": "<extract nationality or N/A>"
  }},
  "education": [
    {{
      "qualification": "<degree or N/A>",
      "fieldOfStudy": "<major or N/A>",
      "instituteName": "<university or N/A>"
    }}
  ],
  "languages": "<extract languages mentioned, comma-separated or N/A>"
}}

Resume:
\"\"\"{resume_text}\"\"\"

Rules:
- If any field is missing, use "N/A"
- Languages must be comma-separated text like English,Hindi,Marathi
- Do not include work experience
- Only extract real values from the resume
"""
    model_instance = genai.GenerativeModel("gemini-1.5-flash")
    response = model_instance.generate_content(
        prompt,
        generation_config=GenerationConfig()
    )

    raw_text = response.text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("Error parsing JSON:", e, "Raw:", cleaned)
        return {}


def format_value(value):
    return value if value is not None else "N/A"
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# jd

def verify_jwt(authorization: str = Header(...)):
    """
    Verify JWT token passed in the Authorization header.
    Expected header: Authorization: Bearer <token>
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Extract raw_text from JWT payload
        raw_text = payload.get("raw_text")
        if not raw_text:
            raise HTTPException(status_code=400, detail="JWT missing 'raw_text'")
        return raw_text
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
class RawJobDescription(BaseModel):
    raw_text: str


PROMPT_TEMPLATE = """
You are a strategic talent attraction specialist and expert copywriter for the aviation and aerospace technology sector. Your writing is clear, concise, and compelling, designed to attract the most innovative and dedicated professionals in the field. Your goal is not just to list duties, but to sell the role, the team, and the company's vision.

**Company Information:**
* **Company Name:** AeroInnovate Solutions
* **Company Mission/Vision:** To revolutionize aerial logistics with autonomous drone technology.
* **Key Company Culture Keywords:** Pioneering, Collaborative, Safety-First, Fast-Paced.

**Instructions:**
1.  Carefully analyze the raw text to extract the core elements of the job.
2.  Follow the structure, tone, and quality demonstrated in the **Gold-Standard Example** below.
3.  Write a powerful opening summary that connects the candidate's potential contribution to the company's ambitious mission.
4.  Rephrase responsibilities and qualifications using dynamic, active language.
5.  Create a 'Why Join AeroInnovate Solutions?' section that highlights the unique value proposition of working at the company.
6.  Conclude with a clear and direct 'Ready to Apply?' call to action.
7.  Format the entire output in clean Markdown.

---
**Gold-Standard Example Output:**

### Avionics Technician at AeroInnovate Solutions

Are you ready to be at the forefront of the autonomous aviation revolution? At AeroInnovate Solutions, we are building the future of logistics, and we are looking for a pioneering Avionics Technician to join our fast-paced team. This isn’t just a job; it’s an opportunity to make a tangible impact on a world-changing technology.

**About The Role**
As our Avionics Technician, you will be the hands-on expert ensuring the reliability and safety of our cutting-edge autonomous drone fleet. You will play a critical role in our mission by maintaining, troubleshooting, and upgrading the very systems that make our vision a reality.

**What You’ll Do**
* Perform comprehensive testing and diagnostics on all avionics systems, including navigation, communication, and control circuits.
* Execute precision installation and integration of new hardware and firmware updates.
* Collaborate closely with our engineering team to provide feedback and drive continuous improvement.
* Maintain meticulous documentation of all maintenance actions in compliance with our safety-first protocols.

**What You’ll Bring**
* A&P License or equivalent certification in electronics/avionics.
* A minimum of 3 years of hands-on experience with complex avionics systems.
* Proven ability to read and interpret schematics and technical manuals.
* A collaborative spirit and a passion for solving complex problems.
* Experience with UAVs or drones is highly desirable.

**Why Join AeroInnovate Solutions?**
* **Be a Pioneer:** Work on technology that is actively shaping the future of an entire industry.
* **Grow With Us:** We invest in our people with continuous training and clear paths for career advancement.
* **Collaborative Culture:** Join a team of brilliant, dedicated professionals who are passionate about our shared mission.

**Ready to Apply?**
If you are driven by innovation and committed to excellence, we want to hear from you. Apply now to help us build the future, today.
---

**Raw Text to Process:**
---
{raw_text}
---
"""
model = genai.GenerativeModel('gemini-1.5-flash')
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
