from pydantic import BaseModel

class JobBase(BaseModel):
    description: str

class JobCreate(JobBase):
    pass

class JobResponse(JobBase):
    id: int
    keywords: str | None

    class Config:
        orm_mode = True
