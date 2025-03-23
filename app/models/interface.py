from pydantic import BaseModel


class SuccessResponse(BaseModel):
    response: str
    message: str

class ChatbotAttributeConfig(BaseModel):
    chatbot_attitude: str
    company_name: str
