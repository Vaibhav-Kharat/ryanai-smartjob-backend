from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from db import get_db
from pydantic import BaseModel
import google.generativeai as genai
from utils import extract_text_from_pdf, decode_jwt, generate_text
from schemas import ResumeParsedResponse
from models import CandidateProfile, Education, User
from services import (
    process_job, verify_token, get_current_user,
    process_candidate, verify_jwt,
    recommend_jobs, get_recommended_candidates_for_job, resume_parsing, format_value, RawJobDescription, PROMPT_TEMPLATE
)
from google.generativeai.types import GenerationConfig
from config import settings
import jwt
import spacy
import re
from datetime import datetime
from logger import gemini_logger  # Import the logger


genai.configure(api_key=settings.GOOGLE_API_KEY)
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
security = HTTPBearer()

# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


@app.post("/process-job")
def process_single_job_route(
    payload: dict = None,
    db: Session = Depends(get_db),
    authorization: str = Header(None)
):
    # ✅ Try extracting job_id from JWT
    decoded = verify_token(authorization) if authorization else None
    job_id = decoded.get("sub") if decoded else None

    # ✅ If not in JWT, fallback to request body
    if not job_id and payload:
        job_id = payload.get("sub") or payload.get("job_id")

    if not job_id:
        raise HTTPException(
            status_code=400, detail="job_id is required (via JWT or body)")

    job = process_job(job_id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job.id, "keywords": job.keywords}
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#


@app.post("/process-candidate")
def process_candidate_route(
    payload: dict = None,
    db: Session = Depends(get_db),
    authorization: str = Header(None)
):
    candidate_id = None

    # 1️⃣ Try extracting candidate_id from JWT
    decoded = verify_token(authorization) if authorization else None
    if decoded:
        candidate_id = decoded.get("sub")

    # 2️⃣ If not in JWT, fallback to request body
    if not candidate_id and payload:
        candidate_id = payload.get("sub") or payload.get("candidate_id")

    # 3️⃣ If still not found → error
    if not candidate_id:
        raise HTTPException(
            status_code=400, detail="candidate_id is required (via JWT or body)")

    # 4️⃣ Process candidate
    candidate = process_candidate(int(candidate_id), db)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {
        "candidate_id": candidate.id,
        "keywords": candidate.keywords
    }


# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
@app.get("/recommend-jobs")
def recommend_jobs_endpoint(
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user)  # <-- JWT-protected
):
    # Use current_user_id instead of hardcoding candidate_id
    candidate_id = current_user_id

    results = recommend_jobs(candidate_id, db)
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")

    return {"candidate_id": candidate_id, "recommended_jobs": results}

# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# here the job id will come through the jwt token


@app.get("/recommend-candidates")
def recommend_candidates_endpoint(request: Request, db: Session = Depends(get_db)):
    # 1️⃣ Extract JWT token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]

    # 2️⃣ Decode JWT to get job_id
    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM])
        job_id = payload.get("job_id")
        if not job_id:
            raise HTTPException(
                status_code=400, detail="Job ID missing in token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 3️⃣ Fetch recommended candidates
    recommended = get_recommended_candidates_for_job(job_id, db)
    if not recommended:
        raise HTTPException(status_code=404, detail="No candidates found")

    return {"job_id": job_id, "recommended_candidates": recommended}


# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
# here get api will be there
@app.get("/parse-resume")
def parse_resume_test(request: Request, db: Session = Depends(get_db)):
    # 1️⃣ Extract JWT token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]

    # 2️⃣ Decode JWT to get candidate_id
    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM])
        candidate_id = payload.get("candidate_id")
        if not candidate_id:
            raise HTTPException(
                status_code=400, detail="Candidate ID missing in token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # --- Fetch candidate ---
    candidate = db.query(CandidateProfile).filter(
        CandidateProfile.id == candidate_id).first()
    if not candidate or not candidate.resumeUrl:
        raise HTTPException(
            status_code=404, detail="Candidate or Resume not found")

    # --- Extract and parse resume ---
    resume_text = extract_text_from_pdf(candidate.resumeUrl)
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

    # --- Improved language parsing ---
    languages_raw = parsed_data.get("languages", "")
    languages_clean = []

    if languages_raw:
        # Remove braces, quotes, or any weird characters
        cleaned = re.sub(r"[{}\[\]\"]", "", languages_raw)
        # Split by commas, strip spaces, ignore empty entries
        languages_clean = [lang.strip()
                           for lang in cleaned.split(",") if lang.strip()]

    # Default to ["N/A"] if nothing found
    candidate.languagesKnown = languages_clean if languages_clean else ["N/A"]

    # --- Save education ---
    db.query(Education).filter(
        Education.candidateProfileId == candidate.id).delete()
    education_saved = []
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
        education_saved.append({
            "qualification": education_entry.qualification,
            "fieldOfStudy": education_entry.fieldOfStudy,
            "instituteName": education_entry.instituteName
        })

    db.commit()

    # --- Return parsed and saved data ---
    return {
        "message": "Resume parsed and data saved successfully",
        "saved_data": {
            "fullName": user.fullName if user else None,
            "phone": candidate.phone,
            "currentLocation": candidate.currentLocation,
            "nationality": candidate.nationality,
            "languagesKnown": candidate.languagesKnown,
            "education": education_saved
        }
    }


@app.get("/enhance-jd")
async def enhance_job_description(raw_text: str = Depends(verify_jwt)):
    if not raw_text or len(raw_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Input text too short")
    full_prompt = PROMPT_TEMPLATE.format(raw_text=raw_text)
    try:
        model_instance = genai.GenerativeModel("gemini-1.5-flash")
        response = model_instance.generate_content(
            full_prompt,
            generation_config=GenerationConfig()
        )

        # Extract usage metadata using the helper method
        usage_metadata = gemini_logger.extract_usage_metadata(response)

        # ✅ Log API call with tokens
        gemini_logger.log_api_call(
            endpoint="enhance_job_description",
            request_data={"prompt_length": len(full_prompt)},
            response_data={
                "usage_metadata": usage_metadata,
                "model": "gemini-1.5-flash",
                "timestamp": datetime.now().isoformat()
            }
        )

        enhanced_jd = response.text
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service failed: {e}")
    return {"status": "success", "enhancedJobDescription": enhanced_jd}
