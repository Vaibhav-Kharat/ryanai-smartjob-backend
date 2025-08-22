import re, json
import io
import google.generativeai as genai
import requests  
from docx import Document 
from pdfminer.high_level import extract_text
from config import settings
from sqlalchemy.orm import Session

genai.configure(api_key=settings.GOOGLE_API_KEY)

def extract_job_keywords(description: str):
    prompt = f"""
    From the job description, extract:
    - Skills: list, comma separated
    - Experience: minimum years or range (e.g., "3-5", "2")

    Return JSON:
    {{
        "skills": ["skill1", "skill2", ...],
        "experience": "3-5"
    }}

    Job Description:
    \"\"\"{description}\"\"\""""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except:
        return {"skills": [], "experience": ""}
    


from sqlalchemy import text

def process_jobs_for_employer(user_id: str, db: Session):
    # Step 1: Get employer id from EmployerProfile
    employer = db.execute(
        text('SELECT id FROM "EmployerProfile" WHERE "userId" = :uid'),
        {"uid": user_id}
    ).mappings().first()

    if not employer:
        print("Employer not found")
        return None

    employer_id = employer["id"]

    # Step 2: Get jobs for this employer
    jobs = db.execute(
        text('SELECT id, description FROM "Job" WHERE "employerId" = :eid'),
        {"eid": employer_id}
    ).mappings().all()

    if not jobs:
        print("No jobs found for this employer")
        return None

    print(f"Found {len(jobs)} job(s) for employer {employer_id}")

    # Step 3: Extract keywords for each job
    processed = []
    for job in jobs:
        extracted = extract_job_keywords(job["description"])
        db.execute(
            text('UPDATE "Job" SET keywords = :kw WHERE id = :jid'),
            {"kw": json.dumps(extracted), "jid": job["id"]}
        )
        processed.append({"job_id": job["id"], "keywords": extracted})

    db.commit()
    return processed








#########################################candidate utils#########################################
def extract_text_from_resume(resume_url: str) -> str:
    """Download and extract text from resume (PDF or DOCX)."""
    response = requests.get(resume_url, timeout=20)
    if response.status_code != 200:
        raise Exception("Could not download resume")

    ext = resume_url.split(".")[-1].lower()
    content = response.content

    if ext == "pdf":
        return extract_text(io.BytesIO(content))
    elif ext == "docx":
        doc = Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise Exception("Unsupported file format")


def extract_resume_keywords(resume_text: str):
    prompt = f"""
    From this resume, extract:
    - Skills (comma separated list)
    - Experience (years or range)

    Return JSON:
    {{
      "skills": ["skill1", "skill2"],
      "experience": "3-5"
    }}

    Resume:
    \"\"\"{resume_text}\"\"\"
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except:
        return {"skills": [], "experience": ""}