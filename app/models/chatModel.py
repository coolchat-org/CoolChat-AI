from typing import List, Optional, Dict, Any, Union
from fastapi import HTTPException, status
from pydantic import Field, field_validator
from beanie import Document, PydanticObjectId
import datetime
from langchain.schema import AIMessage, HumanMessage

class ChatModel(Document):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    conversations: List[List[str]] = Field(default_factory=list)
    is_active: bool = Field(default=True)
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    @field_validator("conversations", mode="before")
    def validate_conversations(cls, value):
        if not isinstance(value, list):
            raise ValueError("Conversations must be a list of lists.")
        if len(value) == 0:
            return value
        for item in value:
            if not (len(item) == 2 and all(isinstance(i, str) for i in item)):
                raise ValueError("Each conversation must be a list containing exactly two strings.")
        return value

    @classmethod
    async def init_session(cls) -> "ChatModel":
        """
        Initialize new chat session and add it to database.
        """
        try:
            new_session = cls()
            await new_session.insert()  
            return new_session
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )

    @classmethod
    async def get_session(cls, session_id: str) -> Optional["ChatModel"]:
        """
        Find session by ID ID.
        """
        try:
            session = await cls.get(PydanticObjectId(session_id))
            return session
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving session: {str(e)}"
            )

    async def save_session(self) -> "ChatModel":
        """
        Update chat session in DB, with updated_at updated.
        """
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            await self.save() 
            return self
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving session: {str(e)}"
            )

    async def update_memory(self, new_messages: List[Dict[str, Any]]) -> "ChatModel":
        """
        Update chat memory with new memory
        """
        self.memory = new_messages
        return await self.save_session()

    async def append_message(self, role: str, content: str) -> "ChatModel":
        """
        Append pair of message to conversation.
        """
        self.conversations.append([role, content])
        return await self.save_session()

    async def close_session(self) -> "ChatModel":
        """
        Close Chat.
        """
        self.is_active = False
        return await self.save_session()

    @classmethod
    def convert_to_langchain_history(cls, memory: List[Dict]) -> List[Union[AIMessage, HumanMessage]]:
        """
        Chuyển đổi memory sang định dạng langchain history.
        """
        return [
            AIMessage(**msg) if msg.get("type") == "ai" 
            else HumanMessage(**msg)
            for msg in memory
        ]

    @classmethod
    def convert_from_langchain_history(cls, messages: List[Union[AIMessage, HumanMessage]]) -> List[Dict]:
        """
        Chuyển đổi từ langchain history sang định dạng memory lưu trong database.
        """
        return [
            {
                "type": "ai" if isinstance(msg, AIMessage) else "human",
                "content": msg.content,
                **msg.additional_kwargs
            }
            for msg in messages
        ]
