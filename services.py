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


def recommend_jobs_for_candidate(candidate_id: int, db: Session):
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == candidate_id).first()
    if not candidate or not candidate.keywords:
        return []

    candidate_skills = set(candidate.keywords.get("skills", []))
    candidate_tools = set(candidate.keywords.get("tools", []))

    # Fetch jobs
    jobs = (
        db.query(Job)
        .options(
            load_only(
                Job.id, Job.salaryMin, Job.salaryMax, Job.title, Job.type,
                Job.location, Job.status, Job.description, Job.vacancies, Job.keywords, Job.categoryId
            ),
            joinedload(Job.employer).load_only(EmployerProfile.companyName),
            joinedload(Job.category).load_only(Category.name)
        )
        .all()
    )

    # Separate category vs keyword jobs
    category_jobs, keyword_jobs = [], []

    for job in jobs:
        job_keywords = job.keywords or {}
        job_skills = set(job_keywords.get("skills", []))
        job_tools = set(job_keywords.get("tools", []))
        match_score = len(candidate_skills.intersection(
            job_skills)) + len(candidate_tools.intersection(job_tools))

        job_data = {
            "id": job.id,
            "salaryMin": job.salaryMin,
            "salaryMax": job.salaryMax,
            "title": job.title,
            "type": job.type,
            "location": job.location,
            "status": job.status,
            "description": job.description,
            "vacancies": job.vacancies,
            "employer": {"companyName": job.employer.companyName if job.employer else None},
            "category": {"name": job.category.name if job.category else None},
            "match_score": match_score
        }

        if job.categoryId == candidate.categoryId:
            category_jobs.append(job_data)
        else:
            keyword_jobs.append(job_data)

    category_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    keyword_jobs.sort(key=lambda x: x["match_score"], reverse=True)

    # Return top 2 category jobs + top 3 keyword jobs
    return category_jobs[:2] + keyword_jobs[:3]


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

    Education:
    - Year of Graduation
    - Grade
    - Institute Name

    Work Experience:
    - Total Experience (in years)
    - Total Flight Hours

    Additional Details:
    - Languages Known (comma separated)

    Return JSON in this format:
    {{
        "full_name": "...",
        "phone": "...",
        "location": "...",
        "nationality": "...",
        "graduation_year": "...",
        "grade": "...",
        "institute": "...",
        "total_experience": "...",
        "total_flight_hours": "...",
        "languages": "..."
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
