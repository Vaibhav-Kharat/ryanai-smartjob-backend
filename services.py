import json
import re
import os
import io
import requests
import fitz  # PyMuPDF
import httpx

# Frontend Callback Configuration:
# To disable frontend callbacks, set environment variable: ENABLE_FRONTEND_CALLBACKS=false
# To change frontend URL, set: FRONTEND_CALLBACK_URL=http://your-frontend-url

import traceback
import google.generativeai as genai
import spacy
from fastapi import HTTPException, Depends, Header
from jose import jwt, JWTError, ExpiredSignatureError
from sqlalchemy.orm import joinedload, load_only
from sqlalchemy.orm import Session, joinedload
from models import Job, CandidateProfile, EmployerProfile, Category, User, CandidateBookmark
from utils import extract_job_keywords, generate_match_reason, generate_text
from db import SessionLocal
from config import settings
from datetime import datetime
from logger import gemini_logger  # Import the logger
# This is the correct class name
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from typing import Optional
from rapidfuzz import fuzz
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
nlp = spacy.load("en_core_web_sm")
genai.configure(api_key=settings.GOOGLE_API_KEY)
load_dotenv()  # Load .env variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


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
            status_code=401, detail="Authorization token missing"
        )

    if token.startswith("Bearer "):
        token = token.split(" ")[1].strip()

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=401, detail="Invalid token payload"
            )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# the below logic is for alert to the candidates that this job is matching to you
async def recommend_candidates_for_job(job_id: str, db: Session):
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job or not job.keywords:
            return

        # Collect candidate data
        candidates = db.query(CandidateProfile).all()
        candidate_data = []
        for c in candidates:
            if not c.keywords:
                continue
            candidate_data.append({
                "candidate_id": c.id,
                "skills": c.keywords.get("skills", []),
                "experience": c.keywords.get("experience", {})
            })

        # Build AI prompt
        prompt = f"""
        You are a job-matching AI.
        Match the job requirements with candidates and recommend the most suitable ones.

        Job Keywords:
        {json.dumps(job.keywords, indent=2)}

        Candidates:
        {json.dumps(candidate_data, indent=2)}

        Return JSON in this format:
        {{
            "recommended": [
                {{
                    "candidate_id": <id>    
                }}
            ]
        }}
        """

        # Call Gemini
        ai_response = generate_text(
            prompt, model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

        try:
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", ai_response.strip())
            cleaned = re.sub(r"\n```$", "", cleaned)
            result = json.loads(cleaned)
        except Exception as e:
            print("AI JSON parse error:", e, "RAW:", ai_response)
            result = {"recommended": []}

        # ✅ Optional frontend callback (can be disabled via environment variable)
        if os.getenv("ENABLE_FRONTEND_CALLBACKS", "true").lower() == "true":
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    FRONTEND_CALLBACK_URL = os.getenv(
                        "FRONTEND_CALLBACK_URL",
                        "http://localhost:8087/recommendations-callback"
                    )

                    response = await client.post(
                        FRONTEND_CALLBACK_URL,
                        json={
                            "job_id": job_id,
                            "recommended_candidates": result.get("recommended", [])
                        }
                    )
                    if response.status_code == 200:
                        print(f"✅ Sent candidate recommendations to frontend for job {job_id}")
            except (httpx.ConnectError, httpx.TimeoutException):
                print(f"ℹ️ Frontend not available - candidate recommendations processed successfully")
            except Exception as e:
                print(f"⚠️ Frontend callback failed (non-critical): {str(e)[:100]}")

    except Exception as e:
        print(f"Error recommending candidates for job {job_id}: {e}")

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


# def process_candidate(candidate_id: int, db: Session):
#     candidate = db.query(CandidateProfile).filter(
#         CandidateProfile.id == candidate_id).first()
#     if not candidate:
#         return None

#     # 1. Extract text from resume
#     resume_text = extract_text_from_resume(candidate.resumeUrl)

#     # 2. Extract skills/experience using Gemini
#     extracted = extract_resume_keywords(resume_text)

#     # 3. Save in DB
#     candidate.keywords = extracted
#     db.commit()
#     db.refresh(candidate)

#     return candidate
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
def get_candidate_id_from_token(authorization: str = Header(...)):
    """
    Extract candidate_id from JWT token in Authorization header
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        candidate_id = payload.get("profileId")
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
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


def decode_jwt_token_recommed_job(token: str):
    if token.startswith("Bearer "):
        token = token.split(" ")[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def safe_int(value, default=0):
    """Safely convert to int, return default if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


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

    candidate_exp_dict = candidate_keywords.get("experience") or {}
    if isinstance(candidate_exp_dict, str):
        try:
            candidate_exp_dict = json.loads(candidate_exp_dict)
        except json.JSONDecodeError:
            candidate_exp_dict = {}

    candidate_skills = set(candidate_keywords.get("skills", []))
    candidate_exp = safe_int(candidate_exp_dict.get("years"))

    # Optimized database query - load only needed fields and relationships
    # Filter out jobs without keywords and limit result size for faster processing
    jobs = db.query(Job).options(
        joinedload(Job.employer),
        joinedload(Job.category)
    ).filter(
        Job.keywords.isnot(None)
    ).limit(100).all()  # Limit to 100 most recent jobs for faster processing
    
    job_matches = []

    # Fast processing with pre-filtering and optimized template-based match reasons
    for job in jobs:
        job_keywords = job.keywords or {}
        if isinstance(job_keywords, str):
            try:
                job_keywords = json.loads(job_keywords)
            except json.JSONDecodeError:
                continue  # Skip jobs with invalid keywords

        job_exp_dict = job_keywords.get("experience") or {}
        if isinstance(job_exp_dict, str):
            try:
                job_exp_dict = json.loads(job_exp_dict)
            except json.JSONDecodeError:
                job_exp_dict = {}

        job_skills = set(job_keywords.get("skills", []))
        if not job_skills:  # Skip jobs with no skills
            continue
        
        # Quick pre-filter: Skip jobs with zero skill overlap
        if not candidate_skills.intersection(job_skills):
            continue
            
        job_min_exp = safe_int(job_exp_dict.get("min_experience"))
        job_max_exp = safe_int(job_exp_dict.get("max_experience"))

        # --- Skill Match ---
        skill_match_pct = (
            (len(candidate_skills.intersection(job_skills)) / len(job_skills)) * 100
            if job_skills else 0
        )

        # --- Experience Match ---
        if job_max_exp == 0:  # no exp specified
            experience_match_pct = 0
        else:
            if job_min_exp <= candidate_exp <= job_max_exp:
                experience_match_pct = 100
            else:
                diff = min(
                    abs(candidate_exp - job_min_exp),
                    abs(candidate_exp - job_max_exp)
                )
                experience_match_pct = max(0, 100 - diff * 20)

        aggregate_pct = (skill_match_pct + experience_match_pct) / 2
        
        # Fast template-based match reason
        if skill_match_pct >= 80:
            reason = f"Excellent skill alignment ({int(skill_match_pct)}%) with strong experience match."
        elif skill_match_pct >= 60:
            reason = f"Good skill match ({int(skill_match_pct)}%) with relevant experience background."
        elif skill_match_pct >= 40:
            reason = f"Moderate skill overlap ({int(skill_match_pct)}%) - good learning opportunity."
        elif skill_match_pct >= 20:
            reason = f"Some transferable skills ({int(skill_match_pct)}%) - training may be needed."
        else:
            reason = f"Entry-level opportunity ({int(skill_match_pct)}% match) - great for career growth."

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
            "job_upsell": job.job_upsell
        })

    # Sort and return top 5
    job_matches.sort(
        key=lambda x: (-x["aggregate_match_percentage"], not x["job_upsell"])
    )

    return job_matches[:5]

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# down logic is for recommend candidates


def recommend_candidates_logic(job_id: str, employer_id: str, db: Session):
    # 1️⃣ Verify employer exists
    employer = db.query(EmployerProfile).filter(
        EmployerProfile.id == employer_id
    ).first()
    if not employer:
        return []

    # 2️⃣ Get job details and parse keywords
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job or not job.keywords:
        return []

    # Parse job keywords safely
    try:
        job_keywords = json.loads(job.keywords) if isinstance(
            job.keywords, str) else job.keywords
    except json.JSONDecodeError:
        job_keywords = {}

    # Handle both dictionary and list formats for job keywords
    if isinstance(job_keywords, dict):
        job_skills = job_keywords.get("skills", [])
        job_exp_dict = job_keywords.get("experience") or {}
        # Convert experience values to integers
        job_min_exp = safe_int_convert(job_exp_dict.get("min_experience", 0))
        job_max_exp = safe_int_convert(job_exp_dict.get("max_experience", 0))
    elif isinstance(job_keywords, list):
        job_skills = job_keywords
        job_min_exp = 0
        job_max_exp = 0
    else:
        job_skills = []
        job_min_exp = 0
        job_max_exp = 0

    # 3️⃣ Fetch all candidates with their related data
    candidates = db.query(CandidateProfile, User, Category).join(
        User, CandidateProfile.userId == User.id
    ).join(
        Category, CandidateProfile.categoryId == Category.id
    ).all()

    # Pre-fetch bookmarks for efficiency
    bookmarks = db.query(CandidateBookmark).filter_by(
        employerId=employer_id).all()
    bookmarked_candidate_ids = {b.candidateId for b in bookmarks}

    recommended = []

    # 4️⃣ Process each candidate
    for candidate, user, category in candidates:
        # Parse candidate keywords safely
        try:
            candidate_keywords = json.loads(candidate.keywords) if isinstance(
                candidate.keywords, str) else candidate.keywords
        except json.JSONDecodeError:
            candidate_keywords = {}

        # Handle both dictionary and list formats for candidate keywords
        if isinstance(candidate_keywords, dict):
            candidate_skills = candidate_keywords.get("skills", [])
            candidate_exp_dict = candidate_keywords.get("experience") or {}
            # Convert candidate experience to integer
            candidate_exp = safe_int_convert(
                candidate_exp_dict.get("years", 0))
        elif isinstance(candidate_keywords, list):
            candidate_skills = candidate_keywords
            candidate_exp = 0  # Default experience if not available
        else:
            candidate_skills = []
            candidate_exp = 0

        # Calculate skill match using fuzzy matching
        skill_match_pct = calculate_skill_match_fuzzy(
            job_skills, candidate_skills)

        # Calculate experience match
        if job_max_exp == 0:  # No max experience specified
            experience_match_pct = 100 if candidate_exp >= job_min_exp else max(
                0, 100 - (job_min_exp - candidate_exp) * 20)
        else:
            if job_min_exp <= candidate_exp <= job_max_exp:
                experience_match_pct = 100
            else:
                diff = min(abs(candidate_exp - job_min_exp),
                           abs(candidate_exp - job_max_exp))
                experience_match_pct = max(0, 100 - diff * 20)

        # Check bookmark status
        is_bookmarked = candidate.id in bookmarked_candidate_ids

        # Build candidate profile
        recommended.append({
            "id": candidate.id,
            "isBookmarked": is_bookmarked,
            "user": user.fullName,
            "openToWork": candidate.openToWork,
            "currentLocation": candidate.currentLocation,
            "aircraftTypeRated": candidate.aircraftTypeRated or [],
            "totalExperience": candidate.totalExperience,
            "noticePeriod": candidate.noticePeriod,
            "preferredJobType": candidate.preferredJobType,
            "skillMatch": round(skill_match_pct, 2),
            "experienceMatch": round(experience_match_pct, 2),
        })

    # 5️⃣ Sort and return top candidates
    recommended.sort(key=lambda x: (
        x["openToWork"], x["skillMatch"]), reverse=True)
    return recommended[:5]


def safe_int_convert(value, default=0):
    """Safely convert a value to integer, returning default if conversion fails"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def calculate_skill_match_fuzzy(job_skills, candidate_skills, threshold=70):
    """Calculate skill match percentage using fuzzy string matching"""
    if not job_skills:
        return 0

    matches = 0
    for js in job_skills:
        for cs in candidate_skills:
            if cs and js and fuzz.token_set_ratio(js.lower(), cs.lower()) >= threshold:
                matches += 1
                break
    return (matches / len(job_skills)) * 100

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


def decode_jwt_token(authorization: str):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        candidate_id = payload.get("profileId")
        if not candidate_id:
            raise HTTPException(
                status_code=401, detail="candidate_id missing in token")
        return candidate_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# --- Resume Text Extractor ---
def extract_text_from_pdf(pdf_url: str) -> str:
    text = ""
    response = requests.get(pdf_url, timeout=20)
    response.raise_for_status()
    pdf_bytes = io.BytesIO(response.content)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


# --- Unified Gemini Parser ---
def unified_resume_parser(resume_text: str):
    prompt = f"""
    Extract all structured details from this resume and return JSON strictly in this format:

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
        "languages": "{{English,Hindi,Marathi}}",
        "skills": ["string"],
        "experience": {{
            "years": "string"
        }}
    }}

    Rules:
    - Do not include work experience descriptions, only extract "years".
    - For education: Degree → qualification, Major → fieldOfStudy, University → instituteName.
    - Languages must be plain text like {{English,Hindi,Marathi}}, no proficiency levels.
    - Skills must be a clean list of technical & professional keywords.
    - Strictly return valid JSON (no markdown, no explanations).

    Resume:
    \"\"\"{resume_text}\"\"\"
    """

    model_instance = genai.GenerativeModel("gemini-1.5-flash")
    response = model_instance.generate_content(
        prompt,
        generation_config=GenerationConfig()
    )

    # Log Gemini usage
    usage_metadata = gemini_logger.extract_usage_metadata(response)
    gemini_logger.log_api_call(
        endpoint="unified_resume_parser",
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
        return json.loads(cleaned)
    except Exception as e:
        print("Error parsing JSON:", e, "Raw:", cleaned)
        return {}


# --- Format Helper ---
def format_value(val):
    return val if val and str(val).strip().lower() != "null" else None



def run_recommendation_task(candidate_id: int):
    db = SessionLocal()
    try:
        candidate = db.query(CandidateProfile).filter(
            CandidateProfile.id == candidate_id
        ).first()

        if not candidate or not candidate.keywords:
            return

        candidate_keywords = candidate.keywords.get("skills", [])

        jobs = db.query(Job).all()
        jobs_data = [
            {
                "jobId": job.id,
                "title": job.title,
                "keywords": job.keywords.get("skills", [])
            }
            for job in jobs
        ]

        # --- AI Prompt for Gemini ---
        prompt = f"""
        You are a job recommendation AI.
        Given a candidate with skills: {candidate_keywords}
        and the following jobs with their required skills:

        {json.dumps(jobs_data, indent=2)}

        Select the most suitable jobs for this candidate based on
        skill match, relevance, and context.

        Return strictly in JSON format like this:
        [
            {{
                "jobId": 1
            }}
        ]
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
            recommended_jobs = json.loads(cleaned)
        except Exception as e:
            print("AI parsing error:", e, "Raw:", cleaned)
            recommended_jobs = []

        # --- Optional frontend callback (can be disabled via environment variable) ---
        if recommended_jobs and os.getenv("ENABLE_FRONTEND_CALLBACKS", "true").lower() == "true":
            try:
                callback_url = os.getenv(
                    "FRONTEND_CALLBACK_URL", 
                    "http://localhost:8087/recommendations-callback-parse-resume"
                )
                payload = {
                    "candidateId": candidate.id,
                    "recommendedJobs": recommended_jobs
                }
                response = requests.post(callback_url, json=payload, timeout=3)
                if response.status_code == 200:
                    print(f"✅ Sent job recommendations to frontend for candidate {candidate.id}")
            except requests.exceptions.ConnectionError:
                print(f"ℹ️ Frontend not available - job recommendations processed successfully")
            except requests.exceptions.Timeout:
                print(f"⚠️ Frontend callback timeout - continuing without notification")
            except Exception as e:
                print(f"⚠️ Frontend callback failed (non-critical): {str(e)[:100]}")

    finally:
        db.close()

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
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

    profile_id = payload.get("raw_text")
    if not profile_id:
        raise HTTPException(status_code=401, detail="Token missing profileId")

    return profile_id
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


def calculate_match_score(candidate_keywords: dict, job_keywords: dict, candidate_experience: int):
    # --- Skills matching ---
    candidate_skills = set(
        map(str.lower, candidate_keywords.get("skills", [])))
    job_skills = set(map(str.lower, job_keywords.get("skills", [])))

    matched_skills = candidate_skills.intersection(job_skills)
    skills_percentage = (len(matched_skills) /
                         len(job_skills)) * 100 if job_skills else 0

    # --- Experience matching ---
    exp_match = 0
    job_exp = job_keywords.get("experience", {})
    job_min_exp = 0
    job_max_exp = 0

    if isinstance(job_exp, dict):
        job_required = job_exp.get("years") or job_exp.get("min_experience")
        job_min_exp = safe_int(job_exp.get("min_experience", 0))
        job_max_exp = safe_int(job_exp.get("max_experience", 0))
        if job_required:
            job_required_num = extract_numeric_experience(str(job_required))
            if candidate_experience and job_required_num:
                exp_match = min(candidate_experience /
                                job_required_num, 1.0) * 100

    # --- Aggregate ---
    aggregate = (skills_percentage + exp_match) / 2

    # --- Match reason (AI-generated with correct parameters) ---
    match_reason = generate_match_reason(
        candidate_experience, candidate_skills, job_min_exp, job_max_exp, job_skills)

    return {
        "skill_match_percentage": round(skills_percentage, 2),
        "experience_match_percentage": round(exp_match, 2),
        "aggregate_match_percentage": round(aggregate, 2),
        "match_reason": match_reason
    }


def extract_numeric_experience(exp_str: str) -> int:
    """
    Convert '4+' or '2-5' or '3 years' into a numeric value
    """
    import re
    match = re.findall(r"\d+", exp_str)
    if not match:
        return None
    return int(match[0])


# Removed duplicate generate_match_reason function - using the one from utils.py instead
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#