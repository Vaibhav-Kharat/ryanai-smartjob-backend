from fastapi import FastAPI, Depends, HTTPException, Request, Header, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from db import get_db
from pydantic import BaseModel
import google.generativeai as genai
# from utils import extract_text_from_pdf, decode_jwt, generate_text
from schemas import ResumeParsedResponse
from models import CandidateProfile, Education, User, EmployerProfile, Job, Category
from services import (
    process_job,
    recommend_jobs_logic, verify_jwt_and_role, get_candidate_id_from_token, decode_jwt_token_job, decode_jwt_token_recommed_job, recommend_candidates_logic, format_value, extract_text_from_pdf, unified_resume_parser, decode_jwt_token, calculate_match_score, recommend_candidates_for_job, run_recommendation_task
)
# This is the correct class name
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from config import settings
import jwt
import spacy
import re
import os
from datetime import datetime
from logger import gemini_logger  # Import the logger


# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
genai.configure(api_key=settings.GOOGLE_API_KEY)
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
security = HTTPBearer()
load_dotenv()  # Load .env variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


# process_single_job_route
@app.get("/process-job/{job_id}")
async def process_single_job_route(
    job_id: str,
    authorization: str = Header(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    if not authorization:
        raise HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    # 🔑 Pass the full Authorization header (with Bearer) to background task
    background_tasks.add_task(
        recommend_candidates_for_job, job_id, authorization)

    job = process_job(job_id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"success": True, "job_id": job.id, "keywords": job.keywords}


# for timming this endpoint
@app.post("/recommendations-callback")
async def recommendations_callback(request: Request):
    data = await request.json()
    print("✅ Received recommendation callback:", data)
    return {"message": "Callback received successfully"}

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


# @app.get("/process-candidate")
# def process_candidate_route(
#     candidate_id: int = Depends(get_candidate_id_from_token),
#     db: Session = Depends(get_db)
# ):
#     candidate = process_candidate(int(candidate_id), db)
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")

#     return {"candidate_id": candidate.id, "keywords": candidate.keywords}
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
@app.get("/recommended-jobs")
def get_recommend_jobs(Authorization: str = Header(...), db: Session = Depends(get_db)):
    start_time = datetime.now()

    # Decode token and extract candidate_id
    payload = decode_jwt_token_recommed_job(Authorization)
    employer_user_id = payload.get("profileId")
    if not employer_user_id:
        raise HTTPException(
            status_code=400, detail="candidate_id not found in token") 

    results = recommend_jobs_logic(employer_user_id, db)
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")

    # Simple completion logging
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(
        f"✅ Job recommendations completed in {processing_time:.2f}s for candidate {employer_user_id}")

    return {"candidate_id": employer_user_id, "recommended_jobs": results}
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


@app.get("/recommended-candidates/{jobId}")
def recommend_candidates(jobId: str, request: Request, db: Session = Depends(get_db)):
    # 1️⃣ Extract and decode JWT
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        employer_id = payload.get("profileId")
        print(f"Decoded JWT payload: {payload}")  # For debugging
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not employer_id:
        raise HTTPException(
            status_code=400, detail="Employer ID missing in token")

    # 2️⃣ Verify job belongs to this employer
    job = db.query(Job).filter(Job.id == jobId).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Ensure consistent type comparison (both as strings or both as integers)
    if str(job.employerId) != str(employer_id):
        raise HTTPException(
            status_code=403, detail="This job does not belong to you")

    # 3️⃣ Get recommended candidates
    recommended_candidates = recommend_candidates_logic(jobId, employer_id, db)
    if not recommended_candidates:
        raise HTTPException(
            status_code=404, detail="No eligible candidates found")

    return {
        "job_id": jobId,
        "recommended_candidates": recommended_candidates
    }
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


@app.get("/parse-resume")
def process_resume(
    db: Session = Depends(get_db),
    authorization: str = Header(...),
    background_tasks: BackgroundTasks = None
):
    # --- Store raw token ---
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    raw_token = authorization.split(" ")[1]  # keep the token as-is

    # --- Still decode for candidate_id ---
    jwt_payload = jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
    candidate_id = jwt_payload.get("profileId")
    if not candidate_id:
        raise HTTPException(
            status_code=401, detail="profileId missing in token")

    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == candidate_id
    ).first()

    if not candidate or not candidate.resumeUrl:
        raise HTTPException(
            status_code=404, detail="Candidate or Resume not found"
        )

    # --- Extract + Parse Resume ---
    resume_text = extract_text_from_pdf(candidate.resumeUrl)
    parsed_data = unified_resume_parser(resume_text)
    if not parsed_data:
        raise HTTPException(status_code=500, detail="Failed to parse resume")

    # --- Save details (same as before) ---
    # ... your save logic ...
    

    if "personalDetails" in parsed_data:
        if candidate.user:
            candidate.user.fullName = format_value(parsed_data["personalDetails"].get("fullName"))
        candidate.phone = format_value(parsed_data["personalDetails"].get("phone"))
        candidate.currentLocation = format_value(parsed_data["personalDetails"].get("currentLocation"))
        candidate.nationality = format_value(parsed_data["personalDetails"].get("nationality"))

    if "education" in parsed_data:
        candidate.education = parsed_data["education"]

    if "languages" in parsed_data:
        candidate.languagesKnown = parsed_data["languages"]

    if "skills" in parsed_data:
        candidate.keywords = {"skills": parsed_data["skills"]}

    if "experience" in parsed_data:
        candidate.experienceYears = format_value(parsed_data["experience"].get("years"))

    db.commit()
    db.refresh(candidate)

    # ✅ Pass raw token to background task
    background_tasks.add_task(run_recommendation_task, candidate.id, raw_token)

    return {
        "message": "Resume processed successfully",
        "candidate_id": candidate.id,
        "keywords": candidate.keywords,
        "personalDetails": parsed_data.get("personalDetails", {}),
        "languages": candidate.languagesKnown,
    }


# for timming this endpoint
@app.post("/recommendations-callback-parse-resume")
async def recommendations_callback(request: Request):
    data = await request.json()
    print("✅ Received recommendation callback:", data)
    return {"message": "Callback received successfully"}

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# jd


class RawJobDescription(BaseModel):
    raw_text: str


# --- Updated Prompt Template ---
# The prompt is now adjusted to work from a single block of text.
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


@app.post("/enhance-jd")
async def enhance_job_description(
    jd_input: RawJobDescription,
    user_id: str = Depends(verify_jwt_and_role)
):
    if not jd_input.raw_text or len(jd_input.raw_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Input text too short")

    full_prompt = PROMPT_TEMPLATE.format(raw_text=jd_input.raw_text)

    try:
        model_instance = genai.GenerativeModel("gemini-1.5-flash")
        response = model_instance.generate_content(
            full_prompt,
            generation_config=GenerationConfig()
        )

        usage_metadata = gemini_logger.extract_usage_metadata(response)
        gemini_logger.log_api_call(
            endpoint="enhance_job_description",
            request_data={"prompt_length": len(full_prompt)},
            response_data={
                "usage_metadata": usage_metadata,
                "model": "gemini-1.5-flash",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
        )

        enhanced_jd = response.text
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service failed: {e}")

    return {"status": "success", "userId": user_id, "enhancedJobDescription": enhanced_jd}
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#


@app.get("/match-score/{jobId}")
def match_score(jobId: str, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    # ✅ Extract profileId from JWT
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    profile_id = payload.get("profileId")
    if not profile_id:
        raise HTTPException(
            status_code=400, detail="Profile ID missing in token")

    # ✅ Fetch candidate and job data
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == profile_id).first()
    job = db.query(Job).filter(Job.id == jobId).first()

    if not candidate or not job:
        raise HTTPException(
            status_code=404, detail="Candidate or Job not found")

    candidate_keywords = candidate.keywords or {}
    job_keywords = job.keywords or {}

    # ✅ Calculate scores
    result = calculate_match_score(
        candidate_keywords, job_keywords, candidate.totalExperience)

    return result
