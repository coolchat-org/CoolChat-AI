from typing import Any, Optional
from pydantic import BaseModel, field_validator
from app.models.interface import ChatbotAttributeConfig

class ChatUserDto(BaseModel):
    new_message: str
    index_host: str
    namespace: str
    virtual_namespace: Optional[str] = None

    config: Optional[ChatbotAttributeConfig]

    # validate: it is string type and not empty
    @field_validator('new_message', mode='before')
    @classmethod
    def ensure_valid_str(cls, v: Any) -> Any:
        if not isinstance(v, str):
            raise ValueError("new_message must be a string")
        if not v.strip():
            raise ValueError("new_message must not be empty")
        return v


class ChatDemoDto(BaseModel):
    message: str
    # index_name: str
    current_index_url: str
    preview_index_url: str

class LLMResponseDto(BaseModel):
    reply: str
    context: Any
    message: str
class LLMResponseDto(BaseModel):
    reply: str
    context: Any
    message: str