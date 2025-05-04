from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.db import mongo_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init DB
    await mongo_connection.connect()

    # yield app
    yield

    # Terminate DB
    await mongo_connection.close()