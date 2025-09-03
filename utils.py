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
import PyPDF2
from config import settings
from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import gemini_logger  # Import the logger
from datetime import datetime
# This is the correct class name
from google.generativeai.types import GenerationConfig


import google.generativeai as genai
from config import settings  # assuming you store your API key in settings

# Set your API key globally
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Example usage


def generate_text(prompt, model="gemini-1.5-flash", temperature=0.7, max_tokens=300):
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    except Exception as e:
        print(f"Error generating text: {e}")
        return None


def extract_job_keywords(description: str):
    prompt = f"""
    From this job description, extract:
    - Skills: list
    - Experience: min & max years 
    Return JSON.
    
    Job Description:
    \"\"\"{description}\"\"\""""
    model_instance = genai.GenerativeModel("gemini-1.5-flash")
    response = model_instance.generate_content(
        prompt,
        generation_config=GenerationConfig()
    )

    usage_metadata = gemini_logger.extract_usage_metadata(response)

    gemini_logger.log_api_call(
        endpoint="extract_job_keywords",
        request_data={"prompt_length": len(prompt)},
        response_data={
            "usage_metadata": usage_metadata,
            "model": "gemini-1.5-flash",
            "timestamp": datetime.now().isoformat()
        }
    )

    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        data = json.loads(cleaned)

        # normalize keys to lowercase
        normalized = {k.lower(): v for k, v in data.items()}

        return {
            "skills": normalized.get("skills", []),
            "experience": {
                "min_experience": normalized.get("experience", {}).get("min"),
                "max_experience": normalized.get("experience", {}).get("max"),
            }
        }
    except Exception as e:
        print("JSON parse error:", e, "RAW:", cleaned)
        return {"skills": [], "experience": {"min_experience": None, "max_experience": None}}


######################################### candidate utils#########################################
def extract_text_from_resume(resume_url: str) -> str:
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
    - Skills
    - Experience in years
    
    Return JSON.
    Resume:
    \"\"\"{resume_text}\"\"\""""
    model_instance = genai.GenerativeModel("gemini-1.5-flash")
    response = model_instance.generate_content(
        prompt,
        generation_config=GenerationConfig()
    )

    # Extract usage metadata using the helper method
    usage_metadata = gemini_logger.extract_usage_metadata(response)

    gemini_logger.log_api_call(
        endpoint="extract_resume_keywords",
        request_data={"prompt_length": len(prompt)},
        response_data={
            "usage_metadata": usage_metadata,
            "model": "gemini-1.5-flash",
            "timestamp": datetime.now().isoformat()
        }
    )

    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return {
            "skills": data.get("skills", []),
            "experience": {"years": data.get("experience")}
        }
    except:
        return {"skills": [], "experience": {"years": None}}


# resume parsing

def extract_text_from_pdf(pdf_url: str) -> str:
  """
  Extracts text from a PDF file.

  Args:
    pdf_path: The path to the PDF file.

  Returns:
    A string containing the extracted text, or None if an error occurred.
  """
  text = ""
  try:
    with fitz.open(pdf_url) as doc:
      for page in doc:
        text += page.get_text()
  except Exception as e:
    print(f"Error extracting text from {pdf_url}: {e}")
    return None
  return text


def decode_jwt(token: str):
    import jwt
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except Exception as e:
        print("JWT Decode Error:", e)
        return None

# this below is the function for why thsis job is a good match for the candidate it is implemented in recommedate job for candidate in services.py


def generate_match_reason(candidate_exp, candidate_skills, job_min_exp, job_max_exp, job_skills):
    prompt = f"""
    Candidate has {candidate_exp} years of experience and skills: {list(candidate_skills)}.
    The job requires {job_min_exp}-{job_max_exp} years of experience and skills: {list(job_skills)}.

    Write a short explanation (2 sentences max) on why this job is a good match for the candidate.
    """

    try:
        model_instance = genai.GenerativeModel("gemini-1.5-flash")
        response = model_instance.generate_content(
            prompt,
            generation_config=GenerationConfig(max_output_tokens=80)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating match reason: {e}")
        return "Explanation could not be generated."
