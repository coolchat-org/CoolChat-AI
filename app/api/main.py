from fastapi import APIRouter
from app.api.v2 import chats as chats_v2

api_router = APIRouter()
# api_router.include_router(chats.v1_router)
api_router.include_router(chats_v2.v2_router)


# if settings.ENVIRONMENT == "local":
#     api_router.include_router(private.router)
