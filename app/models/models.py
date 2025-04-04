from typing import Any
from bson import ObjectId
from pydantic import BaseModel, Field

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate 
    
    @classmethod
    def validate(cls, v: Any, values: Any = None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema: dict):
        # field_schema.update(type="string")
        field_schema.update(type="string")
        return field_schema

class MongooseModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id") # map _id field from mongodb    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}