"""
Configuration management using pydantic-settings
Load from environment variables or .env file
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    groq_api_key: str
    openai_api_key: Optional[str] = None
    
    # Model Configuration
    model_name: str = "mixtral-8x7b-32768"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # File Processing Limits
    max_file_size_mb: int = 10
    max_files_per_request: int = 10
    
    # Processing Configuration
    enable_parallel_processing: bool = True
    timeout_seconds: int = 30
    
    # Feature Flags
    use_ocr: bool = False
    enable_caching: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Usage in main.py:
# from config import get_settings
# settings = get_settings()
# llm_client = GroqLLMClient(api_key=settings.groq_api_key, model=settings.model_name)