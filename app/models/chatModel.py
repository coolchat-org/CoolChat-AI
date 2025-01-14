from typing import ClassVar, List, Tuple, Optional
from bson import ObjectId
from fastapi import HTTPException, status
from pydantic import field_validator
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.models import MongooseModel, PyObjectId
from app.core.db import mongo_connection

class ChatModel(MongooseModel):
    memory: Optional[str] = None # it is optional
    conversations: List[Tuple[str, str]]

    @field_validator("conversations", mode="before")
    def validate_conversations(cls, value):        
        if not isinstance(value, list):
            raise ValueError("Conversations must be a list of tuples.")
        
        if len(value) == 0:
            return value
        for item in value:
            if not (len(item) == 2 and all(isinstance(i, str) for i in item)):
                raise ValueError("Each conversation must be a tuple containing exactly two strings.")
        return value
    
    @classmethod
    async def find_by_id(cls, chat_id: str, collection: AsyncIOMotorCollection) -> Optional["ChatModel"]:
        """Find a chat document by ID using Motor."""
        chat = await collection.find_one({"_id": PyObjectId(chat_id)})
        if chat:
            return cls(**chat)
        return None

    async def save(self, collection: AsyncIOMotorCollection) -> "ChatModel":
        """Save the updated chat document using Motor."""
        result = await collection.replace_one({"_id": self.id}, self.model_dump(by_alias=True))
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Failed to save the updated chat."
            )
        return self
