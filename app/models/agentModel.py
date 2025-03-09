from typing import Annotated, Any, Dict, TypedDict, Optional, List
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel


# Define state schema
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    current_message: str
    should_create_transaction: bool
    transaction_data: Dict[str, Any] | None
    is_end_conversation: bool

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Client gửi session_id để tiếp tục hội thoại
    org_db_host: Optional[str] = None

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatResponse(BaseModel):
    response: str
    is_end: bool = False
    session_id: str  # Trả về session_id cho client