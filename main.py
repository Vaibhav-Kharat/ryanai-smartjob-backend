from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from db import get_db
from pydantic import BaseModel
import google.generativeai as genai
from utils import extract_text_from_pdf, decode_jwt, generate_text
from schemas import ResumeParsedResponse
from models import CandidateProfile, Education, User, EmployerProfile, Job, Category
from services import (
    process_job,
    process_candidate,
    recommend_jobs_logic, resume_parsing, format_value, decode_jwt_token, verify_jwt,get_candidate_id_from_token, decode_jwt_token_job,decode_jwt_token_recommed_candidate, recommend_candidates_logic
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


genai.configure(api_key=settings.GOOGLE_API_KEY)
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
security = HTTPBearer()
load_dotenv()  # Load .env variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


@app.get("/process-job")
def process_single_job_route(
    authorization: str = Header(...),  # expects Authorization header
    db: Session = Depends(get_db)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    payload = decode_jwt_token_job(authorization)
    if not payload:
        raise HTTPException(status_code=401, detail="Failed to decode token")

    # Extract job_id from JWT payload
    job_id = payload.get("job_id") or payload.get("sub")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id not found in token")

    job = process_job(job_id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job.id, "keywords": job.keywords}


@app.get("/process-candidate")
def process_candidate_route(
    candidate_id: int = Depends(get_candidate_id_from_token),
    db: Session = Depends(get_db)
):
    candidate = process_candidate(int(candidate_id), db)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {"candidate_id": candidate.id, "keywords": candidate.keywords}


@app.get("/recommend-jobs")
def get_recommend_jobs(Authorization: str = Header(...), db: Session = Depends(get_db)):
    # Decode token and extract candidate_id
    payload = decode_jwt_token_recommed_candidate(Authorization)
    candidate_id = payload.get("candidate_id")
    if not candidate_id:
        raise HTTPException(status_code=400, detail="candidate_id not found in token")

    results = recommend_jobs_logic(candidate_id, db)
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")

    return {"candidate_id": candidate_id, "recommended_jobs": results}


@app.get("/recommend-candidates")
def recommend_candidates(request: Request, db: Session = Depends(get_db)):
    # 1️⃣ Extract JWT token from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]

    # 2️⃣ Decode JWT to get job_id and employer_user_id
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        job_id = payload.get("job_id")
        employer_user_id = payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not job_id:
        raise HTTPException(status_code=400, detail="Job ID missing in token")

    # 3️⃣ Call the business logic
    recommended_candidates = recommend_candidates_logic(job_id, employer_user_id, db)

    if not recommended_candidates:
        raise HTTPException(status_code=404, detail="No eligible candidates found")

    return {"job_id": job_id, "recommended_candidates": recommended_candidates}




@app.get("/parse-resume")
def parse_resume_test(
    db: Session = Depends(get_db),
    authorization: str = Header(...)
):
    # --- Decode JWT ---
    candidate_id = decode_jwt_token(authorization)

    # --- Fetch Candidate ---
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == candidate_id
    ).first()

    if not candidate or not candidate.resumeUrl:
        raise HTTPException(status_code=404, detail="Candidate or Resume not found")

    # --- Extract resume text ---
    resume_text = extract_text_from_pdf(candidate.resumeUrl)

    # --- Parse resume ---
    parsed_data = resume_parsing(resume_text)
    if not parsed_data:
        raise HTTPException(status_code=500, detail="Failed to parse resume")

    # --- Save personal details ---
    personal = parsed_data.get("personalDetails", {})
    user = db.query(User).filter(User.id == candidate.userId).first()
    if user:
        user.fullName = format_value(personal.get("fullName"))

    candidate.phone = format_value(personal.get("phone"))
    candidate.currentLocation = format_value(personal.get("currentLocation"))
    candidate.nationality = format_value(personal.get("nationality"))

    # --- Save languages ---
    languages_str = parsed_data.get("languages", "")
    candidate.languagesKnown = format_value(
        languages_str.strip("{}") if languages_str else None
    )

    # --- Save education ---
    db.query(Education).filter(
        Education.candidateProfileId == candidate.id
    ).delete()

    for edu in parsed_data.get("education", []):
        education_entry = Education(
            candidateProfileId=candidate.id,
            qualification=format_value(edu.get("qualification")),
            fieldOfStudy=format_value(edu.get("fieldOfStudy")),
            instituteName=format_value(edu.get("instituteName")),
            yearOfGraduation=None,
            grade=None,
            updatedAt=datetime.utcnow(),
            createdAt=datetime.utcnow()
        )
        db.add(education_entry)

    db.commit()
    return {"message": "Resume parsed and data saved successfully"}


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
    jd_input: RawJobDescription, profile_id: str = Depends(verify_jwt)
):
    # ✅ You have validated profile_id from JWT
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
                "profile_id": profile_id  # log the profile using this request
            }
        )

        enhanced_jd = response.text
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service failed: {e}")

    return {"status": "success", "profileId": profile_id, "enhancedJobDescription": enhanced_jd}
