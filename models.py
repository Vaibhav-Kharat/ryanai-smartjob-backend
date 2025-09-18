from sqlalchemy import Column, Text, ForeignKey, Integer, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from db import Base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# -------------------
# Categories
# -------------------


class Category(Base):
    __tablename__ = "Category"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False)

# -------------------
# Users
# -------------------


class User(Base):
    __tablename__ = "User"
    id = Column(Text, primary_key=True, index=True)
    fullName = Column(Text, nullable=True)
    image = Column(Text, nullable=True)
    email = Column(Text, unique=True, index=True)
    password = Column(Text, nullable=False)
    role = Column(Text, nullable=False)

    # optional relationship to CandidateProfile
    candidate_profile = relationship(
        "CandidateProfile", back_populates="user", uselist=False)

# -------------------
# Employers
# -------------------


class EmployerProfile(Base):
    __tablename__ = "EmployerProfile"
    id = Column(Text, primary_key=True, index=True)
    companyName = Column(Text, nullable=False)
    userId = Column(Text, ForeignKey("User.id"), unique=True, nullable=False)
    companyLogo = Column(Text, nullable=True)

    user = relationship("User", backref="employer_profile", uselist=False)

# -------------------
# Jobs
# -------------------


class Job(Base):
    __tablename__ = "Job"
    id = Column(Text, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    keywords = Column(JSONB)
    employerId = Column(Text, ForeignKey("EmployerProfile.id"))
    categoryId = Column(Integer, ForeignKey("Category.id"))
    salaryMin = Column(Integer, nullable=True)
    salaryMax = Column(Integer, nullable=True)
    type = Column(Text, nullable=True)
    location = Column(Text, nullable=True)
    status = Column(Text, nullable=True)
    vacancies = Column(Integer, nullable=True)
    job_upsell = Column(Boolean, default=False)
    slug = Column(Text, nullable=True)
    createdAt = Column(DateTime(timezone=True),
                       server_default=func.now(), nullable=False)
    updatedAt = Column(DateTime(timezone=True), server_default=func.now(
    ), onupdate=func.now(), nullable=False)

    employer = relationship("EmployerProfile", backref="jobs")
    category = relationship("Category", backref="jobs")

# -------------------
# Candidate Profiles (for both resume & parsed info)
# -------------------


class CandidateProfile(Base):
    __tablename__ = "CandidateProfile"
    __table_args__ = {'extend_existing': True}  # avoids InvalidRequestError

    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Text, ForeignKey("User.id"), unique=True, nullable=False)
    resumeUrl = Column(Text, nullable=True)
    keywords = Column(JSONB, nullable=True)
    categoryId = Column(Integer, ForeignKey("Category.id"))
    currentLocation = Column(Text, nullable=True)
    totalExperience = Column(Integer, nullable=True)
    nationality = Column(Text, nullable=True)
    phone = Column(Text, nullable=True)
    languagesKnown = Column(ARRAY(Text), nullable=True)
    openToWork = Column(Boolean, default=False)
    aircraftTypeRated = Column(ARRAY(Text), nullable=True)
    noticePeriod = Column(Text, nullable=True)
    preferredJobType = Column(Text, nullable=True)

    # Relationships
    category = relationship("Category", backref="candidates")
    user = relationship("User", back_populates="candidate_profile")


class Education(Base):
    __tablename__ = "Education"
    id = Column(Integer, primary_key=True, index=True)
    candidateId = Column(Integer, ForeignKey("CandidateProfile.id"))
    qualification = Column(Text, nullable=True)
    fieldOfStudy = Column(Text, nullable=True)   # ✅ make nullable
    instituteName = Column(Text, nullable=True)
    yearOfGraduation = Column(Integer, nullable=True)
    grade = Column(Text, nullable=True)
    createdAt = Column(DateTime(timezone=True),
                       server_default=func.now(), nullable=False)
    updatedAt = Column(DateTime(timezone=True),
                       server_default=func.now(), onupdate=func.now())

    candidate = relationship("CandidateProfile", backref="educations")


class CandidateBookmark(Base):
    __tablename__ = "CandidateBookmark"
    id = Column(Integer, primary_key=True, index=True)
    employerId = Column(Integer, ForeignKey("EmployerProfile.id"))
    candidateId = Column(Integer, ForeignKey("CandidateProfile.id"))


class JobBookmark(Base):
    __tablename__ = "JobBookmark"
    id = Column(Integer, primary_key=True, index=True)
    candidateId = Column(Integer, ForeignKey("CandidateProfile.id"))
    jobId = Column(Text, ForeignKey("Job.id"))


class Application(Base):
    __tablename__ = "Application"

    id = Column(Integer, primary_key=True, index=True)
    jobId = Column(Text, ForeignKey("Job.id"))
    candidateId = Column(Integer, ForeignKey("CandidateProfile.id"))
    status=Column(Text)

    job = relationship("Job", backref="applications")
    candidate = relationship("CandidateProfile", backref="applications")
