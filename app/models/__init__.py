from typing import List, Tuple, Optional
from pydantic import field_validator

from app.models.models import MongooseModel

class ChatModel(MongooseModel):
    memory: Optional[str] = None # it is optional
    conversations: List[Tuple[str, str]]

    @field_validator("conversations", mode="before")
    def validate_conversations(cls, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Each conversation must be a tuple of exactly two strings.")
        return value