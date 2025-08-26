from pydantic import BaseModel
from typing import Optional

class JobBase(BaseModel):
    description: str

class JobCreate(JobBase):
    pass

class JobResponse(JobBase):
    id: int
    keywords: str | None

    class Config:
        orm_mode = True


class ResumeParseRequest(BaseModel):
    candidate_id: int

class ResumeParsedResponse(BaseModel):
    full_name: Optional[str]
    phone: Optional[str]
    location: Optional[str]
    nationality: Optional[str]
    graduation_year: Optional[str]
    grade: Optional[str]
    institute: Optional[str]
    total_experience: Optional[str]
    total_flight_hours: Optional[str]
    languages: Optional[str]