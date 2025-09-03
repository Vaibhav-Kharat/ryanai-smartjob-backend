import json
import re
import os

import traceback
import google.generativeai as genai
import spacy
from fastapi import HTTPException, Depends, Header
from jose import jwt, JWTError, ExpiredSignatureError
from sqlalchemy.orm import joinedload, load_only
from sqlalchemy.orm import Session, joinedload
from models import Job, CandidateProfile, EmployerProfile, Category, User
from utils import extract_job_keywords, extract_text_from_resume, extract_resume_keywords, generate_match_reason
from config import settings
from datetime import datetime
from logger import gemini_logger  # Import the logger
# This is the correct class name
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from typing import Optional
from rapidfuzz import fuzz


nlp = spacy.load("en_core_web_sm")
genai.configure(api_key=settings.GOOGLE_API_KEY)
load_dotenv()  # Load .env variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


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


def decode_jwt_token_job(token: str):
    if not token:
        raise HTTPException(
            status_code=401, detail="Authorization header missing")

    if token.startswith("Bearer "):
        token = token.split(" ")[1].strip()

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=401, detail="Invalid token payload")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
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


def get_candidate_id_from_token(authorization: str = Header(...)):
    """
    Extract candidate_id from JWT token in Authorization header
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        candidate_id = payload.get("candidate_id")
        if not candidate_id:
            raise HTTPException(
                status_code=401, detail="candidate_id missing in token")
        return candidate_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def normalize_skills(skills):
    if not skills:
        return set()
    if isinstance(skills, str):  # candidate case: comma-separated string
        return set([s.strip() for s in skills.split(",") if s.strip()])
    if isinstance(skills, list):  # job case
        return set([s.strip() for s in skills if isinstance(s, str) and s.strip()])
    return set()


def normalize_experience(exp):
    if not exp:
        return None
    if isinstance(exp, str):  # e.g. "2-4"
        return exp.strip()
    return str(exp).strip()


def recommend_jobs_logic(candidate_id: int, db: Session):
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
        job_keywords = job.keywords or {}
        if isinstance(job_keywords, str):
            try:
                job_keywords = json.loads(job_keywords)
            except json.JSONDecodeError:
                job_keywords = {}

        job_exp_dict = job_keywords.get("experience")
        if job_exp_dict is None:
            job_exp_dict = {}
        elif isinstance(job_exp_dict, str):
            try:
                job_exp_dict = json.loads(job_exp_dict)
            except json.JSONDecodeError:
                job_exp_dict = {}

        job_skills = set(job_keywords.get("skills", []))
        job_min_exp = job_exp_dict.get("min_experience") or 0
        job_max_exp = job_exp_dict.get("max_experience") or 0

        skill_match_pct = (
            (len(candidate_skills.intersection(job_skills)) / len(job_skills)) * 100
            if job_skills else 0
        )

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
            "match_reason": reason,
            "job_upsell": job.job_upsell  # Add upsell flag for sorting
        })

    # Sort by upsell first, then aggregate % (upsell=True first)
    job_matches.sort(
        key=lambda x: (not x["job_upsell"], -x["aggregate_match_percentage"])
    )

    return job_matches[:5]


def decode_jwt_token_recommed_candidate(token: str):
    if token.startswith("Bearer "):
        token = token.split(" ")[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# down logic is for recommend candidates


def recommend_candidates_logic(job_id: str, employer_user_id: str, db: Session):
    # 1️⃣ Verify employer
    employer = db.query(EmployerProfile).filter(
        EmployerProfile.userId == employer_user_id
    ).first()
    if not employer:
        return []

    # 2️⃣ Get job + keywords
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job or not job.keywords:
        return []

    job_keywords = job.keywords
    if isinstance(job_keywords, str):
        try:
            job_keywords = json.loads(job_keywords)
        except json.JSONDecodeError:
            job_keywords = {}

    job_skills = job_keywords.get("skills", [])
    job_exp_dict = job_keywords.get("experience") or {}
    job_min_exp = job_exp_dict.get("min_experience", 0)
    job_max_exp = job_exp_dict.get("max_experience", 0)

    # 3️⃣ Fetch all candidates
    candidates = db.query(CandidateProfile, User, Category).join(
        User, CandidateProfile.userId == User.id
    ).join(
        Category, CandidateProfile.categoryId == Category.id
    ).all()

    recommended = []

    # Fuzzy skill matching function
    def calculate_skill_match_fuzzy(job_skills, candidate_skills, threshold=70):
        matches = 0
        for js in job_skills:
            for cs in candidate_skills:
                if cs and js and fuzz.token_set_ratio(js, cs) >= threshold:
                    matches += 1
                    break
        return (matches / len(job_skills)) * 100 if job_skills else 0

    for candidate, user, category in candidates:
        candidate_keywords = candidate.keywords
        if isinstance(candidate_keywords, str):
            try:
                candidate_keywords = json.loads(candidate_keywords)
            except json.JSONDecodeError:
                candidate_keywords = {}

        candidate_skills = candidate_keywords.get("skills", [])
        candidate_exp_dict = candidate_keywords.get("experience") or {}
        candidate_exp = candidate_exp_dict.get("years") or 0

        # --- Skills match %
        skill_match_pct = calculate_skill_match_fuzzy(job_skills, candidate_skills)

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

        recommended.append({
            "id": candidate.id,
            "fullName": user.fullName,
            "image": user.image,
            "jobCategory": category.name if category else None,
            "currentLocation": candidate.currentLocation,
            "totalExperience": candidate.totalExperience,
            "nationality": candidate.nationality,
            "resumeUrl": candidate.resumeUrl,
            "skill_match_percentage": round(skill_match_pct),
            "experience_match_percentage": round(experience_match_pct),
            "aggregate_match_percentage": round(aggregate_pct),
        })

    # 4️⃣ Sort by aggregate match %
    recommended.sort(key=lambda x: x["aggregate_match_percentage"], reverse=True)
    return recommended[:5]



def resume_parsing(resume_text: str):
    prompt = f"""
    Extract resume details and return JSON in this format only:

    {{
        "personalDetails": {{
            "fullName": "string",
            "phone": "string",
            "currentLocation": "string",
            "nationality": "string"
        }},
        "education": [
            {{
                "qualification": "string",
                "fieldOfStudy": "string",
                "instituteName": "string"
            }}
        ],
        "languages": "{{English,Hindi,Marathi}}"
    }}

    Rules:
    - Remove work experience completely
    - In education, only map Degree → qualification, Major → fieldOfStudy, University → instituteName
    - Languages must be text format like {{English,Hindi,Marathi}}, no levels

    Resume:
    \"\"\"{resume_text}\"\"\""""

    model_instance = genai.GenerativeModel("gemini-1.5-flash")
    response = model_instance.generate_content(
        prompt,
        generation_config=GenerationConfig()
    )

    # Print the full response for debugging
    print("Full response structure:", response)

    # Extract usage metadata using the helper method
    usage_metadata = gemini_logger.extract_usage_metadata(response)
    print("Extracted usage metadata:", usage_metadata)

    # ✅ Log API call with correct usage metadata
    gemini_logger.log_api_call(
        endpoint="extract_with_gemini",
        request_data={"prompt_length": len(prompt)},
        response_data={
            "usage_metadata": usage_metadata,
            "model": "gemini-1.5-flash",
            "timestamp": datetime.now().isoformat()
        }
    )

    raw_text = response.text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)   # now Gemini output matches final schema
    except Exception as e:
        print("Error parsing JSON:", e, "Raw:", cleaned)
        return {}


# this function i sfor resume parsing when the output value is null we can return N/A

def format_value(value):
    return value if value is not None else "N/A"


def decode_jwt_token(token: str):
    """
    Decode JWT token and extract candidate_id.
    Automatically strips 'Bearer ' if present.
    Logs detailed error information with stack trace.
    """
    # Remove Bearer prefix if present
    if token.startswith("Bearer "):
        token = token.split(" ")[1].strip()

    try:
        print("Decoding token:", token)
        print("Using secret key:", SECRET_KEY)
        print("Algorithm:", ALGORITHM)

        # Decode the token using loaded secret
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("Decoded JWT Payload:", payload)

        candidate_id = payload.get("candidate_id")
        if candidate_id is None:
            raise HTTPException(
                status_code=401, detail="Invalid token: candidate_id missing"
            )

        return candidate_id

    except ExpiredSignatureError as e:
        print("JWT ExpiredSignatureError:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=401, detail="Token has expired")

    except JWTError as e:
        print("JWTError:", str(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=401, detail="Invalid or malformed token")

    except Exception as e:
        print("Unexpected error decoding JWT:", str(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Internal server error during token decoding")


# this is for jd api jwt authentication
def verify_jwt(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    profile_id = payload.get("profileId")
    if not profile_id:
        raise HTTPException(status_code=401, detail="Token missing profileId")

    return profile_id
