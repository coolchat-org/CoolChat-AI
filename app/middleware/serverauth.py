from typing import Awaitable, Callable
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings

class VerifyInternalKeyMiddleware(BaseHTTPMiddleware):
    secret: str = settings.AI_SERVICE_API_KEY
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            return await call_next(request)
        
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {self.secret}":
            raise HTTPException(status_code=401, detail="Unauthorized!")
        
        return await call_next(request)