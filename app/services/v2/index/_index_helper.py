import asyncio
import logging
import tempfile
import os
from typing import Any, Dict, List, Tuple
from charset_normalizer import detect
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.asyncio import tqdm_asyncio
from pinecone import ServerlessSpec
from app.core.config import settings
from app.utils.crawlsite import crawl_sites

async def init_shared_index(pinecone_client) -> None:
    """
    Ensure the shared Pinecone index exists and is ready.
    """
    name = settings.PINECONE_INDEX_NAME
    existing = [idx["name"] for idx in pinecone_client.list_indexes()]
    if name not in existing:
        pinecone_client.create_index(
            name=name,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_REGION),
        )
        logger.info("Creating shared index '%s'...", name)
        # wait until ready
        while not pinecone_client.describe_index(name).status["ready"]:
            await asyncio.sleep(1)
    else:
        logger.info("Shared index '%s' already exists.", name)
