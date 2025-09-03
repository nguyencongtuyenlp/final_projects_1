from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_prefix = "APP_"
        env_file = ".env"

settings = Settings()
