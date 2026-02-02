from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 2000
    enable_llm: bool = True
    
    # Application Configuration
    max_file_size_mb: int = 100
    allowed_extensions: str = "csv,xlsx"
    
    # FastAPI Configuration
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Configuration
    cors_origins: str = '["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"]'
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    project_name: str = "ML Dataset Analyser"
    project_version: str = "1.0.0"
    
    # Helper properties
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from string to list"""
        import json
        try:
            return json.loads(self.cors_origins)
        except json.JSONDecodeError:
            return ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse allowed extensions from string to list"""
        return self.allowed_extensions.split(",")
    
    class Config:
        env_file = ".env"

settings = Settings()
