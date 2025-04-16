from typing import Optional, Dict
from pinecone import Pinecone
from app.core.config import settings
import time
import logging

logger = logging.getLogger(__name__)

class PineconeConnectionPool:
    _instance: Optional['PineconeConnectionPool'] = None
    _client: Optional[Pinecone] = None
    _index_cache: Dict[str, tuple] = {}  # (index, last_accessed_time)
    _cache_ttl = 3600  # 1 hour TTL for cached indexes
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._client = Pinecone(api_key=settings.PINECONE_API_KEY)
            logger.info("Initialized Pinecone client")
    
    def get_index(self, host: str):
        current_time = time.time()
        
        # Check cache
        if host in self._index_cache:
            index, last_accessed = self._index_cache[host]
            if current_time - last_accessed < self._cache_ttl:
                # Update last accessed time
                self._index_cache[host] = (index, current_time)
                return index
        
        # Create new index if not in cache or expired
        try:
            index = self._client.Index(host=host)
            self._index_cache[host] = (index, current_time)
            return index
        except Exception as e:
            logger.error(f"Error creating Pinecone index for host {host}: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear expired indexes from cache"""
        current_time = time.time()
        expired_hosts = [
            host for host, (_, last_accessed) in self._index_cache.items()
            if current_time - last_accessed > self._cache_ttl
        ]
        for host in expired_hosts:
            del self._index_cache[host]
    
    @classmethod
    def get_instance(cls) -> 'PineconeConnectionPool':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance