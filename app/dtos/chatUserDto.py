from typing import Any
from pydantic import BaseModel, field_validator


class ChatUserDto(BaseModel):
    new_message: str
    # validate: it is string type and not empty
    @field_validator('new_message', mode='before')
    @classmethod
    def ensure_valid_str(cls, v: Any) -> Any:
        if not isinstance(v, str):
            raise ValueError("new_message must be a string")
        if not v.strip():
            raise ValueError("new_message must not be empty")
        return v