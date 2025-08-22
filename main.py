from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from db import get_db
from services import (
    process_single_job,
    process_candidate,
    recommend_jobs_for_candidate, recommend_candidates_for_job
)
import jwt
from config import settings

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
    candidate_id = 3  # <-- static candidate id (change as needed for testing)

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
