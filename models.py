from sqlalchemy import Column, Text, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from db import Base
from sqlalchemy.orm import relationship

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
    userId = Column(Text, unique=True, nullable=False)

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

    # Relationships
    category = relationship("Category", backref="candidates")
    user = relationship("User", back_populates="candidate_profile")
