from pydantic import BaseModel


class SuccessResponse(BaseModel):
    response: str
    message: str