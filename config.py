from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    GOOGLE_API_KEY: str
    FRONTEND_CALLBACK_URL_BASE: str

    # --- Constructed URLs (not from env, dynamically built) ---
    @property
    def FRONTEND_CALLBACK_URL_PROCESS_JOB(self) -> str:
        return f"{self.FRONTEND_CALLBACK_URL_BASE}/notify-candidates"

    @property
    def FRONTEND_CALLBACK_URL_PARSE_RESUME(self) -> str:
        return f"{self.FRONTEND_CALLBACK_URL_BASE}/notify-employers"

    class Config:
        env_file = ".env"

settings = Settings()
