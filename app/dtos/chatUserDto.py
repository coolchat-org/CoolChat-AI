from typing import Any, List
from pydantic import BaseModel, field_validator

from typing import List, Union

# Định nghĩa kiểu dữ liệu
RawMessage = List[str]
RawHistory = Union[List[str], List[Union[str, RawMessage]]]

def validate_raw_history(raw_history: RawHistory):
    if not isinstance(raw_history, list):
        raise TypeError("raw_history phải là một danh sách.")
    
    # Kiểm tra phần tử đầu tiên (nếu có) là summary hoặc danh sách tin nhắn
    if isinstance(raw_history[0], str):  # Nếu phần tử đầu tiên là summary
        if not all(isinstance(pair, list) and len(pair) == 2 and 
                   all(isinstance(msg, str) for msg in pair)
                   for pair in raw_history[1:]):
            raise ValueError("Các phần tử sau summary phải là danh sách cặp tin nhắn [str, str].")
    else:  # Nếu không có summary
        if not all(isinstance(pair, list) and len(pair) == 2 and 
                   all(isinstance(msg, str) for msg in pair)
                   for pair in raw_history):
            raise ValueError("Mỗi phần tử trong raw_history phải là danh sách cặp tin nhắn [str, str].")

# Ví dụ sử dụng
# raw_history = ["Summary here", ["hello", "Tôi là chatbot"], ["How are you?", "I'm fine"]]
# validate_raw_history(raw_history)  # Nếu không có lỗi, raw_history hợp lệ


class ChatUserDto(BaseModel):
    new_message: str
    org_db_host: str
    # validate: it is string type and not empty
    @field_validator('new_message', mode='before')
    @classmethod
    def ensure_valid_str(cls, v: Any) -> Any:
        if not isinstance(v, str):
            raise ValueError("new_message must be a string")
        if not v.strip():
            raise ValueError("new_message must not be empty")
        return v
    
class CreateIndexDto(BaseModel):
    message: str
    index_name: str
    index_url: str

class LLMResponseDto(BaseModel):
    reply: str
    context: Any
    message: str