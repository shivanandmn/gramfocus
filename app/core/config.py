from pydantic_settings import BaseSettings
from functools import lru_cache
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"

class TranscriptionProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"

class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "GramFocus"
    API_V1_STR: str = "/api/v1"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB default
    CHUNK_SIZE: int = 1024 * 1024  # 1MB chunks for streaming
    
    # LLM Configuration
    LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    TRANSCRIPTION_PROVIDER: TranscriptionProvider = TranscriptionProvider.OPENAI
    
    # Model Names
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_WHISPER_MODEL: str = "whisper-1"
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Google Cloud Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")  # For Gemini
    GOOGLE_CLOUD_PROJECT: str = ""  # For Speech-to-Text
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # For Speech-to-Text

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
