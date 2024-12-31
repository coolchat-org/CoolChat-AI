from fastapi import APIRouter

from app.api.routes import chats
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(chats.router)


# if settings.ENVIRONMENT == "local":
#     api_router.include_router(private.router)
