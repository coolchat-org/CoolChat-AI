from fastapi import APIRouter

from app.api.v1 import chats
from app.api.v2 import chats as chats_v2
# from app.core.config import settings

api_router = APIRouter()
api_router.include_router(chats.v1_router)
api_router.include_router(chats_v2.v2_router)


# if settings.ENVIRONMENT == "local":
#     api_router.include_router(private.router)
