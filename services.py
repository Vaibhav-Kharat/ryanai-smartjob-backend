import json
import re
import google.generativeai as genai
import spacy

from sqlalchemy.orm import joinedload, load_only
from sqlalchemy.orm import Session, joinedload
from models import Job, CandidateProfile, EmployerProfile, Category, User
from utils import extract_job_keywords, extract_text_from_resume, extract_resume_keywords
from config import settings

nlp = spacy.load("en_core_web_sm")


def process_single_job(job_id: str, db: Session):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return None

    extracted = extract_job_keywords(job.description)

    # ✅ Store in keywords column (JSONB)
    job.keywords = extracted
    db.commit()
    db.refresh(job)

    return job


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


def recommend_jobs_for_candidate(candidate_id: int, db: Session):
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
            "aggregate_match_percentage": round(aggregate_pct)
        })

    # Sort by aggregate %
    job_matches.sort(
        key=lambda x: x["aggregate_match_percentage"], reverse=True)
    # Return top 5
    return job_matches[:5]


def recommend_candidates_for_job(job_id: str, employer_user_id: str, db: Session):
    """
    Recommend candidates for a given job based on job_id and employer_user_id.
    """

    # Get the employer
    employer = (
        db.query(EmployerProfile)
        .filter(EmployerProfile.userId == employer_user_id)
        .first()
    )
    if not employer:
        return {"error": "Employer not found"}

    # Get the job
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return {"error": "Job not found"}

    # Fetch candidates with join to User and Category
    candidates = (
        db.query(CandidateProfile, User, Category)
        .join(User, CandidateProfile.userId == User.id)
        .join(Category, CandidateProfile.categoryId == Category.id)
        .all()
    )

    # Build response
    recommended_candidates = []
    for candidate, user, category in candidates:
        candidate_data = {
            "id": candidate.id,
            "fullName": user.fullName,
            "image": user.image,
            "jobCategory": category.name if category else None,
            "currentLocation": candidate.currentLocation,
            "totalExperience": candidate.totalExperience,
            "nationality": candidate.nationality,
            "resumeUrl": candidate.resumeUrl,
            "isBookmarked": False,   # you can update this logic if bookmarks exist
            "match_score": 0,        # placeholder for matching logic
        }
        recommended_candidates.append(candidate_data)

    return {
        "job_id": job.id,
        "recommended_candidates": recommended_candidates,
    }


def extract_with_gemini(resume_text: str):
    prompt = f"""
    Extract the following fields from the resume:
    Personal Info:
    - Full Name
    - Phone Number
    - Location
    - Nationality (based on country)
    Education: (for each education entry found)
    - Qualification (e.g., "B. Tech")
    - Field of Study (e.g., "CSE")
    - Institute Name
    - Year of Graduation
    - Grade/Score
    - Dates (if available)
    Work Experience:
    - Total Experience (in years)
    - Total Flight Hours
    Additional Details:
    - Languages Known (comma separated)
    
    Return JSON in this format:
    {{
        "fullName": "...",
        "phone": "...",
        "currentLocation": "...",
        "nationality": "...",
        "education": [
            {{
                "qualification": "...",
                "fieldOfStudy": "...",
                "instituteName": "...",
                "yearOfGraduation": "...",
                "grade": "..."
            }},
            ... // more education entries if present
        ],
        "totalExperience": "...",
        "totalFlightHours": "...",
        "languagesKnown": "..."
    }}
    
    Resume:
    \"\"\"{resume_text}\"\"\"
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("Error parsing JSON:", e)
        return {}
