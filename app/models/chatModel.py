from typing import ClassVar, List, Tuple, Optional
from bson import ObjectId
from fastapi import HTTPException, status
from pydantic import Field, field_validator
from motor.motor_asyncio import AsyncIOMotorCollection
from app.dtos.chatUserDto import RawHistory
from app.models.models import MongooseModel, PyObjectId
from app.core.db import mongo_connection

class ChatModel(MongooseModel):
    memory: RawHistory = Field(default_factory=list)
    conversations: List[List[str]] = Field(default_factory=list)
    # conversations: List[Tuple[str, str]]

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
    async def init(cls, collection: AsyncIOMotorCollection) -> "ChatModel":
        """
        Khởi tạo một bản ghi mới với memory và conversations là danh sách rỗng, 
        sau đó lưu bản ghi vào MongoDB.

        Args:
            collection (AsyncIOMotorCollection): Collection MongoDB.

        Returns:
            ChatModel: Đối tượng ChatModel mới tạo.
        """
        try:
            # Tạo bản ghi mới
            new_chat = cls()
            
            # Chèn vào MongoDB
            result = await collection.insert_one(new_chat.model_dump(by_alias=True))
            if not result.inserted_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to insert the new chat document into the database."
                )
            
            # Gắn ID vào bản ghi mới và trả về đối tượng ChatModel
            new_chat.id = result.inserted_id
            return new_chat

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
    
    @classmethod
    async def find_by_id(cls, chat_id: str, collection: AsyncIOMotorCollection) -> Optional["ChatModel"]:
        """Find a chat document by ID using Motor."""
        chat = await collection.find_one({"_id": PyObjectId(chat_id)})
        if chat:
            return cls(**chat)
        return None
    
    async def save(self, collection: AsyncIOMotorCollection) -> "ChatModel":
        """Save the updated chat document using Motor."""
        result = await collection.update_one(
            {"_id": self.id},
            {"$set": self.model_dump(by_alias=True)}
        )
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save the updated chat."
            )
        return self

