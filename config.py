from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    GOOGLE_API_KEY: str
    FRONTEND_CALLBACK_URL_PROCESS_JOB: str 
    FRONTEND_CALLBACK_URL_PARSE_RESUME: str

    class Config:
        env_file = ".env"

settings = Settings()
