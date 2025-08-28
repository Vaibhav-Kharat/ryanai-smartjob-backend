from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from db import get_db
from pydantic import BaseModel
import google.generativeai as genai
from utils import extract_text_from_pdf, decode_jwt
from schemas import ResumeParsedResponse
from models import CandidateProfile
from services import (
    process_single_job,
    process_candidate,
    recommend_jobs_for_candidate, recommend_candidates_for_job, extract_with_gemini
)
from config import settings
import jwt
import spacy
import re
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
security = HTTPBearer()


@app.post("/process_job")
def process_single_job_route(
    payload: dict,  # Expecting {"job_id": "..."}
    db: Session = Depends(get_db)
):
    job_id = payload.get("sub")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")

    job = process_single_job(job_id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job.id, "keywords": job.keywords}


@app.post("/process_candidate")
def process_candidate_route(payload: dict, db: Session = Depends(get_db)):
    candidate_id = payload.get("candidate_id")
    if not candidate_id:
        raise HTTPException(status_code=400, detail="candidate_id is required")

    candidate = process_candidate(int(candidate_id), db)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {"candidate_id": candidate.id, "keywords": candidate.keywords}


@app.get("/recommend_jobs")
def recommend_jobs(db: Session = Depends(get_db)):
    # ⚠️ TEMPORARY: Hardcoded candidate_id until frontend passes correct id
    candidate_id = 14  # <-- static candidate id (change as needed for testing)

    results = recommend_jobs_for_candidate(candidate_id, db)
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")

    return {"candidate_id": candidate_id, "recommended_jobs": results}


# When JWT is Ready (future version)

# @app.get("/recommend_jobs")
# def recommend_jobs(request: Request, db: Session = Depends(get_db)):
#     auth_header = request.headers.get("Authorization")
#     if not auth_header or not auth_header.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Missing or invalid token")

#     token = auth_header.split(" ")[1]
#     try:
#         payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
#         user_id = payload.get("sub")   # ⚠️ right now this is userId, not candidateId
#     except jwt.InvalidTokenError:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     # Later: map userId → candidateId via CandidateProfile.userId
#     candidate = db.query(CandidateProfile).filter(CandidateProfile.userId == user_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")

#     return {"candidate_id": candidate.id, "recommended_jobs": recommend_jobs_for_candidate(candidate.id, db)}


@app.get("/recommend_candidates/{job_id}")
def recommend_candidates(job_id: str, request: Request, db: Session = Depends(get_db)):
    # 1️⃣ Extract JWT token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]

    # 2️⃣ Decode JWT to get employer user ID
    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM])
        employer_user_id = payload.get("sub")  # string ID from JWT
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 3️⃣ Get recommended candidates
    recommended = recommend_candidates_for_job(job_id, employer_user_id, db)
    if not recommended:
        raise HTTPException(status_code=404, detail="No candidates found")

    return {"job_id": job_id, "recommended_candidates": recommended}


# When JWT is Ready (future version)
# @app.get("/parse_resume", response_model=ResumeParsedResponse)
# def parse_resume(request: Request, db: Session = Depends(get_db)):
#     # 1. Get token from header
#     auth_header = request.headers.get("Authorization")
#     if not auth_header or not auth_header.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Missing or invalid token")

#     token = auth_header.split(" ")[1]

#     # 2. Decode JWT to get candidate_id
#     candidate_id = decode_jwt(token)
#     if not candidate_id:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     # 3. Fetch candidate profile (to get resumeUrl)
#     candidate = db.query(CandidateProfiles).filter(
#         CandidateProfiles.userId == candidate_id).first()
#     if not candidate or not candidate.resumeUrl:
#         raise HTTPException(
#             status_code=404, detail="Candidate or Resume not found")

#     # 4. Extract text from resume PDF
#     resume_text = extract_text_from_pdf(candidate.resumeUrl)

#     # 5. Parse with Gemini
#     parsed_data = extract_with_gemini(resume_text)

#     # 6. Just return parsed data (NO DB update)
#     return parsed_data


@app.get("/parse_resume_test/{candidate_id}")
def parse_resume_test(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == str(candidate_id)).first()
    if not candidate or not candidate.resumeUrl:
        raise HTTPException(
            status_code=404, detail="Candidate or Resume not found")

    resume_text = extract_text_from_pdf(candidate.resumeUrl)
    parsed_data = extract_with_gemini(resume_text)
    return parsed_data


# jd
class RawJobDescription(BaseModel):
    raw_text: str


# --- Updated Prompt Template ---
# The prompt is now adjusted to work from a single block of text.
PROMPT_TEMPLATE = """
You are an expert HR copywriter for the aviation industry.
Your task is to take a raw, unstructured job description text and transform it into a professional, well-formatted, and engaging job post.  

**Instructions:**
1.  From the text provided, identify the Job Title, key responsibilities, and required qualifications.
2.  Structure the output with clear headings like 'Position Summary', 'Key Responsibilities', and 'Required Qualifications'.
3.  Rewrite the content in a professional and compelling tone to attract qualified candidates.
4.  Format the responsibilities and qualifications as bullet points.
5.  Ensure the final output is in clean Markdown format.

**Raw Text to Process:**
---
{raw_text}
---
"""
model = genai.GenerativeModel('gemini-1.5-flash')


@app.post("/api/enhance-jd")
async def enhance_job_description(jd_input: RawJobDescription):
    """
    Receives a raw block of text for a job description and enhances it using AI.
    """
    if model is None:
        raise HTTPException(
            status_code=500, detail="AI model not configured correctly.")

    if not jd_input.raw_text or len(jd_input.raw_text.strip()) < 20:
        raise HTTPException(
            status_code=400, detail="Input text is too short to process.")

    # Construct the full prompt
    full_prompt = PROMPT_TEMPLATE.format(raw_text=jd_input.raw_text)

    # Call the Gemini API
    try:
        response = model.generate_content(full_prompt)
        enhanced_jd = response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        raise HTTPException(
            status_code=503, detail="AI service failed to process the request.")

    # Send the successful response
    return {
        "status": "success",
        "enhancedJobDescription": enhanced_jd
    }
