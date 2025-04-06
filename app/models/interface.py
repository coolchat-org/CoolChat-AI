from pydantic import BaseModel


class SuccessResponse(BaseModel):
    response: str
    message: str

class ChatbotAttributeConfig(BaseModel):
    chatbot_attitude: str
    company_name: str
    start_sentence: str
    end_sentence: str
