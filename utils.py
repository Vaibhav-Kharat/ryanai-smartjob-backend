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
from logger import gemini_logger  # Import the logger
from datetime import datetime
# This is the correct class name
from google.generativeai.types import GenerationConfig


import google.generativeai as genai
from config import settings  # assuming you store your API key in settings

# Set your API key globally
genai.configure(api_key=settings.GOOGLE_API_KEY)


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
# def extract_text_from_resume(resume_url: str) -> str:
#     response = requests.get(resume_url, timeout=20)
#     if response.status_code != 200:
#         raise Exception("Could not download resume")

#     ext = resume_url.split(".")[-1].lower()
#     content = response.content

#     if ext == "pdf":
#         return extract_text(io.BytesIO(content))
#     elif ext == "docx":
#         doc = Document(io.BytesIO(content))
#         return "\n".join([p.text for p in doc.paragraphs])
#     else:
#         raise Exception("Unsupported file format")


# def extract_resume_keywords(resume_text: str):
#     prompt = f"""
#     From this resume, extract:
#     - Skills
#     - Experience in years
    
#     Return JSON.
#     Resume:
#     \"\"\"{resume_text}\"\"\""""
#     model_instance = genai.GenerativeModel("gemini-1.5-flash")
#     response = model_instance.generate_content(
#         prompt,
#         generation_config=GenerationConfig()
#     )

#     # Extract usage metadata using the helper method
#     usage_metadata = gemini_logger.extract_usage_metadata(response)

#     gemini_logger.log_api_call(
#         endpoint="extract_resume_keywords",
#         request_data={"prompt_length": len(prompt)},
#         response_data={
#             "usage_metadata": usage_metadata,
#             "model": "gemini-1.5-flash",
#             "timestamp": datetime.now().isoformat()
#         }
#     )

#     cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
#     cleaned = re.sub(r"\n```$", "", cleaned)
#     try:
#         data = json.loads(cleaned)
#         return {
#             "skills": data.get("skills", []),
#             "experience": {"years": data.get("experience")}
#         }
#     except:
#         return {"skills": [], "experience": {"years": None}}


# # resume parsing

# def extract_text_from_pdf(pdf_url: str) -> str:
#     text = ""
#     response = requests.get(pdf_url)
#     response.raise_for_status()
#     pdf_bytes = io.BytesIO(response.content)

#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     for page in doc:
#         text += page.get_text()
#     return text


# def decode_jwt(token: str):
#     import jwt
#     try:
#         payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
#         return payload.get("sub")
#     except Exception as e:
#         print("JWT Decode Error:", e)
#         return None

# this below is the function for why thsis job is a good match for the candidate it is implemented in recommedate job for candidate in services.py


def generate_match_reason(candidate_exp, candidate_skills, job_min_exp, job_max_exp, job_skills,
                          skills_percentage, experience_percentage, aggregate_percentage):
    """
    Always generate a reason through AI, aligned with actual scores.
    """

    try:
        candidate_skills = list(candidate_skills) if candidate_skills else []
        job_skills = list(job_skills) if job_skills else []

        prompt = f"""
        You are an AI recruitment assistant.

        Candidate profile:
        - Experience: {candidate_exp} years
        - Skills: {', '.join(candidate_skills) if candidate_skills else 'None'}

        Job requirements:
        - Experience: {job_min_exp} to {job_max_exp if job_max_exp else '∞'} years
        - Skills: {', '.join(job_skills) if job_skills else 'None'}

        Match scores:
        - Skills Match: {skills_percentage}%
        - Experience Match: {experience_percentage}%
        - Overall Match: {aggregate_percentage}%

        Task:
        Write a clear 20–25 word explanation.
        If scores are very low (<30%), explain why this is a poor match.
        If scores are high (>70%), explain why this is a strong match.
        Be consistent with the scores, don’t contradict them.
        """

        model_instance = genai.GenerativeModel("gemini-1.5-flash")
        response = model_instance.generate_content(
            prompt,
            generation_config=GenerationConfig(
                max_output_tokens=100,
                temperature=0.3
            )
        )

        return response.text.strip()

    except Exception as e:
        print(f"Error generating match reason: {e}")
        return "AI-generated match reasoning unavailable. Please retry later."
