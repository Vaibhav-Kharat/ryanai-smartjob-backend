import re
import json
import io
from io import BytesIO
import jwt
import google.generativeai as genai
import requests
import fitz  # PyMuPDF for extracting text from PDF
from docx import Document
from pdfminer.high_level import extract_text
from config import settings
from sqlalchemy.orm import Session
from sqlalchemy import text

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
    \"\"\"{description}\"\"\"
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        skills = data.get("skills", [])

        exp = data.get("experience", "").strip()
        min_exp, max_exp = None, None

        if exp:
            if "-" in exp:
                parts = exp.split("-")
                min_exp = int(parts[0].strip())
                max_exp = int(parts[1].strip())
            else:
                min_exp = max_exp = int(exp)

        experience_dict = {"min_experience": min_exp,
                           "max_experience": max_exp}

        return {
            "skills": skills,
            "experience": experience_dict
        }

    except:
        return {
            "skills": [],
            "experience": {"min_experience": None, "max_experience": None}
        }


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


######################################### candidate utils#########################################
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
    - Experience in years (as a single number)

    Return JSON:
    {{
      "skills": ["skill1", "skill2"],
      "experience": "3"
    }}

    Resume:
    \"\"\"{resume_text}\"\"\"
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        skills = data.get("skills", [])

        exp = data.get("experience", "").strip()
        exp_years = int(exp) if exp.isdigit() else None

        experience_dict = {"years": exp_years}

        return {
            "skills": skills,
            "experience": experience_dict
        }

    except:
        return {
            "skills": [],
            "experience": {"years": None}
        }


# resume parsing
def extract_text_from_pdf(pdf_url: str) -> str:
    text = ""
    # Download PDF content
    response = requests.get(pdf_url)
    response.raise_for_status()  # raise error if download fails
    pdf_bytes = BytesIO(response.content)

    # Open PDF from bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")  # candidate_id stored in `sub`
    except Exception as e:
        print("JWT Decode Error:", e)
        return None
