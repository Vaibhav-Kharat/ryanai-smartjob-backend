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
    # print(f"Recommended jobs for candidate {employer_user_id}: {results}")
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

    # 2️⃣ Fetch employer from DB using profileId
    employer = db.query(EmployerProfile).filter(
        EmployerProfile.id == payload.get("profileId")
    ).first()
    if not employer:
        raise HTTPException(status_code=404, detail="Employer not found")

    # 3️⃣ Verify job belongs to this employer
    job = db.query(Job).filter(Job.id == jobId).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.employerId != employer.id:
        print(
            f"Job employerId: {job.employerId}, Employer DB ID: {employer.id}")
        raise HTTPException(
            status_code=403, detail="This job does not belong to you")

    # 4️⃣ Get recommended candidates
    recommended_candidates = recommend_candidates_logic(jobId, employer.id, db)
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
            candidate.user.fullName = format_value(
                parsed_data["personalDetails"].get("fullName"))
        candidate.phone = format_value(
            parsed_data["personalDetails"].get("phone"))
        candidate.currentLocation = format_value(
            parsed_data["personalDetails"].get("currentLocation"))
        candidate.nationality = format_value(
            parsed_data["personalDetails"].get("nationality"))

    if "education" in parsed_data:
        # optional: clear old records for candidate
        db.query(Education).filter(
            Education.candidateId == candidate.id).delete()

        for edu in parsed_data["education"]:
            new_edu = Education(
                candidateId=candidate.id,
                qualification=edu.get("qualification"),
                fieldOfStudy=edu.get("fieldOfStudy"),
                instituteName=edu.get("instituteName"),
                updatedAt=datetime.utcnow()
            )
            db.add(new_edu)   # Commit after adding all education entries

    if "languages" in parsed_data:
        candidate.languagesKnown = parsed_data["languages"]

    # 🔹 Store skills + experience together inside keywords
    keywords_data = {}
    if "skills" in parsed_data:
        keywords_data["skills"] = parsed_data["skills"]

    if "experience" in parsed_data and parsed_data["experience"].get("years"):
        keywords_data["experience"] = {
            "years": parsed_data["experience"].get("years")}

    if keywords_data:
        candidate.keywords = keywords_data

    if "experience" in parsed_data:
        candidate.totalExperience = int(
            re.sub(r'\D', '', parsed_data["experience"].get("years", "0")) or 0)

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
        "education": parsed_data.get("education", []),
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
  <role>
    You are a strategic talent attraction specialist and expert copywriter 
    for the aviation and aerospace industry. 
    Your writing is clear, concise, and inspiring — crafted to attract 
    innovative, skilled, and mission-driven professionals across all aviation categories, 
    including engineering, avionics, pilots, UAV, maintenance, operations, safety, R&D, and beyond.  
    Your objective is to transform raw job details into compelling job descriptions 
    that not only outline responsibilities but also sell the role, the team, and the vision.  
  </role>

  <instructions>
    1. Read and analyze the <raw_text> to extract the role’s core elements: title, purpose, responsibilities, and qualifications.  
    2. Write a powerful <opening_summary> that connects the candidate’s potential contribution to the company’s mission in aviation/aerospace.  
    3. Convert responsibilities and qualifications into active, engaging bullet points under <responsibilities> and <qualifications>.  
    4. Emphasize how the role contributes to innovation, safety, performance, or efficiency in aviation.  
    5. Create a <why_join_us> section that highlights growth opportunities, pioneering work, and collaborative culture.  
    6. Conclude with a strong <call_to_action> that directly invites candidates to apply.  
    7. Deliver the final job description in clean Markdown formatting, following the defined <structure>.  
  </instructions>

  <restrictions>
  - Only generate if <raw_text> clearly relates to the aviation or aerospace industry.  
    This includes (but is not limited to) any roles, functions, equipment, operations, 
    or technologies connected to aircraft, airlines, airports, avionics, UAVs, drones, 
    space systems, flight operations, maintenance, safety, or aerospace engineering.  
  - The validation must be semantic, not just keyword-based — i.e., if the text 
    describes a role that is part of aviation/aerospace (e.g., pilot, flight instructor, 
    avionics technician, MRO engineer, air traffic controller, aerospace designer), 
    treat it as valid even if specific keywords vary.  
  - If <raw_text> is unrelated to aviation/aerospace, return this exact plain text 
    (with no formatting or extras):  
    Cannot generate job description. Enter a valid aviation-related text.  
  </restrictions>


  <structure>
  - Job Title → Render as a level 3 Markdown heading (###)  
  - Opening Summary → Plain paragraph text immediately after title  
  - About The Role → Subheading (**About The Role**) followed by paragraph  
  - Responsibilities → Subheading (**What You’ll Do**) with bullet points  
  - Qualifications → Subheading (**What You’ll Bring**) with bullet points  
  - Why Join Us → Subheading (**Why Join Us?**) followed by bullet points or short paragraph  
  - Call to Action → Subheading (**Ready to Apply?**) with direct invitation 
  </structure>


  <validation>
    If <raw_text> does not include aviation/aerospace terms, 
    output only this plain text with no formatting:  
    Cannot generate job description. Enter a valid aviation-related text.  
  </validation>

  <general_discussion>
    Do not engage in any other conversation or topics outside of 
    generating aviation/aerospace job descriptions.  
  </general_discussion>


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
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
