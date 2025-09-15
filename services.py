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
import jwt
from fastapi import HTTPException, Depends, Header
from jose import jwt, JWTError, ExpiredSignatureError
from jwt import InvalidTokenError, ExpiredSignatureError
from sqlalchemy import func
from sqlalchemy.orm import joinedload, load_only
from sqlalchemy.orm import Session, joinedload
from models import Job, CandidateProfile, EmployerProfile, Category, User, CandidateBookmark, JobBookmark, Application
from utils import extract_job_keywords, generate_match_reason, generate_text
from db import SessionLocal, get_db
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
async def recommend_candidates_for_job(job_id: str, authorization: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job or not job.keywords:
            return

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

        # --- AI Prompt ---
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

        ai_response = generate_text(
            prompt, model="gemini-1.5-flash", temperature=0.3, max_tokens=500
        )

        try:
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", ai_response.strip())
            cleaned = re.sub(r"\n```$", "", cleaned)
            result = json.loads(cleaned)
        except Exception as e:
            print("AI JSON parse error:", e, "RAW:", ai_response)
            result = {"recommended": []}

        # --- Build final payload ---
        final_recommendations = []
        for rec in result.get("recommended", []):
            cand = db.query(CandidateProfile).filter(
                CandidateProfile.id == rec.get("candidate_id")
            ).first()
            if cand:
                final_recommendations.append({
                    "candidateId": cand.id,
                    "userId": cand.userId,
                    "title": job.title,
                    "slug": job.slug,
                    "email": cand.user.email,
                    "fullName": cand.user.fullName
                })

        # --- Send to frontend with original JWT ---
        if final_recommendations and os.getenv("ENABLE_FRONTEND_CALLBACKS", "true").lower() == "true":
            try:
                with httpx.Client(timeout=5.0) as client:
                    callback_url = settings.FRONTEND_CALLBACK_URL_PROCESS_JOB
                    headers = {"Authorization": authorization}

                    payload = {
                        "job_id": job_id,
                        "recommended_candidates": final_recommendations
                    }

                    response = client.post(
                        callback_url, json=payload, headers=headers)

                    print("📡 Posting to:", callback_url)
                    print("📩 Headers:", headers)
                    print("📦 Payload:", final_recommendations)
                    print("✅ Callback status:", response.status_code)
                    print("🔎 Callback response:", response.text)

                    if response.status_code == 200:
                        print(
                            f"✅ Sent candidate recommendations to frontend for job {job_id}"
                        )
            except (httpx.ConnectError, httpx.TimeoutException):
                print(
                    "ℹ️ Frontend not available - candidate recommendations processed successfully")
            except Exception as e:
                print(
                    f"⚠️ Frontend callback failed (non-critical): {str(e)[:100]}")
    finally:
        db.close()


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
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        print(f"JWT decode failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


def safe_int(val):
    """Convert various inputs to int safely (returns 0 on failure)."""
    if val is None:
        return 0
    if isinstance(val, int):
        return val
    try:
        # if string like "18" or "18 years" -> extract digits
        s = str(val)
        digits = re.sub(r"\D", "", s)
        return int(digits) if digits else 0
    except Exception:
        return 0

def normalize_skills(skills):
    """
    Ensure skills is a list -> return a normalized set (lowercased, stripped).
    Accepts list, JSON string, or plain string.
    """
    if not skills:
        return set()
    # if skills is a JSON string, try to load it
    if isinstance(skills, str):
        try:
            parsed = json.loads(skills)
            skills = parsed
        except Exception:
            # treat single string as single-element list
            skills = [skills]
    # Now expect iterable of strings
    normalized = set()
    for s in (skills or []):
        if not s:
            continue
        try:
            normalized.add(str(s).strip().lower())
        except Exception:
            continue
    return normalized

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

    # candidate experience may be nested
    candidate_exp_dict = candidate_keywords.get("experience") or {}
    if isinstance(candidate_exp_dict, str):
        try:
            candidate_exp_dict = json.loads(candidate_exp_dict)
        except json.JSONDecodeError:
            candidate_exp_dict = {}

    candidate_skills = normalize_skills(candidate_keywords.get("skills", []))
    candidate_exp = safe_int(candidate_exp_dict.get("years"))

    # Debug: print candidate info
    print(f"\n🔎 Candidate {candidate_id} skills: {candidate_skills}, experience: {candidate_exp}")

    # Optimized database query - load only needed fields and relationships
    # Filter out jobs without keywords and limit result size for faster processing
    jobs = db.query(Job).options(
        joinedload(Job.employer),
        joinedload(Job.category)
    ).filter(
        Job.keywords.isnot(None)
    ).all()

    job_matches = []

    # Fast processing with pre-filtering and optimized template-based match reasons
    for job in jobs:
        job_keywords = job.keywords or {}
        if isinstance(job_keywords, str):
            try:
                job_keywords = json.loads(job_keywords)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping job {job.id} due to invalid keywords JSON")
                continue  # Skip jobs with invalid keywords

        job_exp_dict = job_keywords.get("experience") or {}
        if isinstance(job_exp_dict, str):
            try:
                job_exp_dict = json.loads(job_exp_dict)
            except json.JSONDecodeError:
                job_exp_dict = {}

        job_skills = normalize_skills(job_keywords.get("skills", []))
        if not job_skills:  # Skip jobs with no skills
            # Debug
            print(f"❌ Skipped job {job.id} ({job.title}) — no job skills present")
            continue

        # Quick pre-filter: Skip jobs with zero skill overlap
        overlap = candidate_skills.intersection(job_skills)
        # Debug
        print(f"\n📌 Job {job.id} - {job.title}")
        print(f"Job skills: {job_skills}")
        print(f"Overlap with candidate: {overlap}")

        if not overlap:
            print(f"❌ Skipped job {job.id} — no matching skills")
            continue

        job_min_exp = safe_int(job_exp_dict.get("min_experience"))
        job_max_exp = safe_int(job_exp_dict.get("max_experience"))

        # --- Skill Match ---
        skill_match_pct = (
            (len(overlap) / len(job_skills)) * 100
            if job_skills else 0
        )

        # --- Experience Match ---
        if job_max_exp == 0:
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

        # Debug: print computed percentages
        print(f"Skill%: {skill_match_pct:.1f}, Exp%: {experience_match_pct:.1f}, Agg%: {aggregate_pct:.1f}")

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

        # --- Bookmark, Apply & Applications Count ---
        isBookmarked = db.query(JobBookmark).filter_by(
            candidateId=candidate_id, jobId=job.id
        ).first() is not None

        isApplied = db.query(Application).filter_by(
            candidateId=candidate_id, jobId=job.id
        ).first() is not None

        applicationsCount = db.query(func.count(Application.id)).filter_by(
            jobId=job.id
        ).scalar()

        job_matches.append({
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salaryMin": job.salaryMin,
            "salaryMax": job.salaryMax,
            "employer": {
                "companyName": job.employer.companyName if job.employer else None,
                "companyLogo": job.employer.companyLogo if job.employer else None,
                "id": job.employer.id if job.employer else None,
            },
            "slug": job.slug,
            "type": job.type,
            "isBookmarked": isBookmarked,
            "isApplied": isApplied,
            "applicationsCount": applicationsCount,
            "createdAt": job.createdAt.isoformat() if job.createdAt else None,
            "updatedAt": job.updatedAt.isoformat() if job.updatedAt else None,
            "category": {"name": job.category.name if job.category else None},
            "skill_match_percentage": round(skill_match_pct),
            "experience_match_percentage": round(experience_match_pct),
            "aggregate_match_percentage": round(aggregate_pct),
            "match_reason": reason,
            "job_upsell": job.job_upsell
        })

    # Sort and return top 5 (same as your original logic)
    job_matches.sort(
        key=lambda x: (-x["aggregate_match_percentage"], not x["job_upsell"])
    )

    return job_matches[:5]

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# down logic is for recommend candidates

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
    You are provided with a avaiation industry resume text. Your task is to extract and neatly compile the information into a precise JSON format, strictly adhering to the provided structure and rules.

    Please follow these steps to complete the task:

    Extract the personal details including full name, phone number, current location, and infer nationality from the country mentioned.
    Extract structured education details related to the aviation industry from the resume text provided below. Follow these guidelines to ensure accurate data extraction:

    Qualification: Identify the aviation-related degree, diploma, or certification mentioned (e.g., Bachelor of Aviation, Aeronautical Engineering, Pilot’s License).
    Field of Study: Extract the specific area of study or specialization within aviation (e.g., Aerodynamics, Aviation Management, Air Traffic Control).
    Institute Name: Find the name of the aviation school, college, or training institution where the qualification was obtained (e.g., Embry-Riddle Aeronautical University, FAA Academy).
    Ensure that each educational entry is processed separately to accurately capture each detail.
    Collect languages spoken, ensuring plain text format, separated by commas.
    Compile a list of technical and professional skills from the resume.
    Extract only the number of years of experience from the work history.
    Ensure the JSON is valid and complete with no missing fields. Use the following template and replace placeholders with actual data extracted from the resume:

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
        "languages": ["English", "Hindi", "Marathi"],
        "skills": ["string"],
        "experience": {{
            "years": "string"
        }}
    }}

    Rules:
    - Nationality must be according to the country name (e.g. If country name is India nationality will be  Indian, and if country name is Russia nationality will be Russian).
    - Do not include work experience descriptions, only extract "years".
    - For education: Degree → qualification, Major → fieldOfStudy, University → instituteName.
    - Languages must be returned as a JSON array of strings, e.g. ["English", "Hindi", "Marathi"].
    - Do not return as a single string or character-split values.
    - Skills must be a clean list of technical & professional keywords.
    - Strictly return valid JSON (no markdown, no explanations).
    - Extract structured education details from the resume text provided below. Follow these instructions   carefully for precise data extraction:
    Qualification: Identify the degree or diploma mentioned in the resume (e.g., Bachelor of Science, Master’s, Diploma).
    Field of Study: Extract the main subject or major associated with the qualification (e.g., Computer Science, Business Administration, Electrical Engineering).
    Institute Name: Find the name of the educational institution where the qualification was obtained (e.g., Harvard University, Oxford College).
    Consider each educational entry separately and be thorough in capturing the necessary details for each.

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


def run_recommendation_task(candidate_id: int, authorization: str):
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

        # --- AI Prompt ---
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
                "jobId": "string"
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

        # --- Build final payload ---
        final_recommendations = []
        for rec in recommended_jobs:
            job = db.query(Job).filter(Job.id == rec.get("jobId")).first()
            if job:
                final_recommendations.append({
                    "id": job.id,
                    "employerId": job.employerId,
                    "title": job.title,
                    "candidateId": candidate.id,
                    "userId": job.employer.userId,
                    "slug": job.slug,
                    "employerEmail": job.employer.user.email if job.employer and job.employer.user else None,
                    "employerName": job.employer.user.fullName if job.employer and job.employer.user else None
                })

        # --- Send to frontend with original JWT ---
        if final_recommendations and os.getenv("ENABLE_FRONTEND_CALLBACKS", "true").lower() == "true":
            try:
                callback_url = settings.FRONTEND_CALLBACK_URL_PARSE_RESUME
                payload = {
                    "candidateId": candidate.id,
                    "recommendedJobs": final_recommendations
                }
                headers = {"Authorization": f"Bearer {authorization}"}

                print("📡 Posting to:", callback_url)
                print("📩 Headers:", headers)
                print("📦 Payload:", payload)
                response = requests.post(
                    callback_url, json=payload, headers=headers, timeout=5)
                print("✅ Callback status:", response.status_code)
                print("🔎 Callback response:", response.text)

                if response.status_code == 200:
                    print(
                        f"✅ Sent job recommendations to frontend for candidate {candidate.id}")
                    print("Using token:", authorization)
            except Exception as e:
                print(f"⚠️ Callback failed: {str(e)[:100]}")

    finally:
        db.close()


# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# this is for jd api jwt authentication


def verify_jwt_and_role(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
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

    # ✅ fix key name
    user_id = payload.get("userId")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user_id")

    role = payload.get("role")
    if role.upper() != "EMPLOYER":
        raise HTTPException(
            status_code=403, detail="Only EMPLOYERs can enhance JD")

    # check DB for role
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    print("verify jwt and role user role is ", user.role)

    if user.role.upper() != "EMPLOYER":
        raise HTTPException(
            status_code=403, detail="Only EMPLOYERs can enhance JD")

    return user_id

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
