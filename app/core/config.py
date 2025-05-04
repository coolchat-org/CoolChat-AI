import secrets  # Bổ sung import thư viện secrets
from typing import Annotated, Any, Literal
from pydantic import AnyUrl, BeforeValidator, HttpUrl, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> list[str] | str:
    if v == "*":
        return ["*"]
    elif isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",  # File .env để lấy thông tin cấu hình
        env_ignore_empty=True,
        extra="ignore",
    )

    # General settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str  # Đọc từ .env
    INDEX_NAME: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    FRONTEND_HOST: str = "http://localhost:5173"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    # CORS settings
    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors) 
    ] = []

    @computed_field
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    # Project information
    PROJECT_NAME: str = "FastAPI Chatbots"
    SENTRY_DSN: HttpUrl | None = None

    # MongoDB settings
    MONGO_URI: str
    MONGO_DATABASE: str
    OPENAI_API_KEY: str 
    PINECONE_API_KEY: str
    # HTTP_PROXY: str
    AI_SERVICE_API_KEY: str
    FIRECRAWL_API_KEY: str
    GOOGLE_AI_API_KEY: str


# Khởi tạo settings
settings = Settings()
