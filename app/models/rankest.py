import hashlib
import json
from typing import Any, ClassVar, Dict, List, Tuple, Optional, Set, override
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from pydantic import Field, BaseModel
from cachetools import TTLCache, cached
import asyncio
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from dataclasses import dataclass

class DocumentCache:
    """LRU Cache for document storage with TTL"""
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
    
    def get(self, key: str) -> Optional[Document]:
        return self.cache.get(key)
    
    def set(self, key: str, doc: Document):
        self.cache[key] = doc
    
    def clear(self):
        self.cache.clear()

class PriorityScaler:
    """Efficient priority scaling with caching"""
    @staticmethod
    @lru_cache(maxsize=1000)
    def scale_priorities(priorities_tuple: Tuple[int, ...]) -> List[float]:
        priorities = list(priorities_tuple)
        if not priorities:
            return []
        
        max_prio, min_prio = max(priorities), min(priorities)
        if max_prio == min_prio:
            return [1.0] * len(priorities)
            
        scale = 1.0 / (max_prio - min_prio)
        return [(x - min_prio) * scale for x in priorities]

    

class SearchResult:
    def __init__(self, document: Document, vector_score: float = 0.0, keyword_score: float = 0.0, priority_score: float = 0.0):
        self.document = document
        self.vector_score = vector_score
        self.keyword_score = keyword_score
        self.priority_score = priority_score


class CoolChatVectorStore(PineconeVectorStore):
    """
    Vector store kết hợp 3 thành phần:
      - Semantic (vector) search
      - Keyword (TF-IDF) search
      - Business priority metadata
    """
    def __init__(
        self,
        index,
        embedding,
        text_key: str = "text",
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        min_final_threshold: float = 0.1,
        **kwargs
    ):
        super().__init__(index=index, embedding=embedding, text_key=text_key, **kwargs)
        # Weights
        self.alpha = alpha  # weight for semantic similarity
        self.beta = beta    # weight for keyword match
        self.gamma = gamma  # weight for priority
        self.min_final_threshold = min_final_threshold

        # Cache cho document list và query results
        self.doc_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self.result_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)

        # TF-IDF cho keyword search
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.doc_texts: List[str] = []
        self.doc_mapping: Dict[int, Document] = {}
        self.tfidf_matrix = None

    def _preprocess_text(self, text: str) -> str:
        # Loại bỏ ký tự đặc biệt, giữ dấu tiếng Việt
        text = re.sub(r"[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]", ' ', text)
        return text.lower().strip()
    
    def initialize_documents(self, documents: List[Document]):
        """Nạp tài liệu cho TF-IDF tìm kiếm và scale priority metadata"""
        self.doc_texts.clear()
        self.doc_mapping.clear()
        priorities = []

        for idx, doc in enumerate(documents):
            text = self._preprocess_text(doc.page_content)
            self.doc_texts.append(text)
            self.doc_mapping[idx] = doc
            # Đọc và lưu priority metadata (convert về float)
            try:
                priorities.append(float(doc.metadata.get('priority', 1)))
            except ValueError:
                priorities.append(1.0)

        # Scale priority sang [0,1]
        if priorities:
            min_p, max_p = min(priorities), max(priorities)
            self.priority_scores = [
                (p - min_p) / (max_p - min_p) if max_p > min_p else 1.0
                for p in priorities
            ]
        else:
            self.priority_scores = []

        if self.doc_texts:
            self.tfidf_matrix = self.tfidf.fit_transform(self.doc_texts)

    async def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        if not self.doc_texts or self.tfidf_matrix is None:
            return []
        q = self._preprocess_text(query)
        q_vec = self.tfidf.transform([q])
        sims = (q_vec @ self.tfidf_matrix.T).toarray()[0]
        top_idx = sims.argsort()[-k:][::-1]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score > 0:
                results.append(SearchResult(
                    document=self.doc_mapping[idx],
                    keyword_score=score,
                    priority_score=self.priority_scores[idx]
                ))
        return results
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 3,
        **kwargs
    ) -> List[Document]:
        # Thực hiện song song vector và keyword
        vector_task = asyncio.create_task(
            self.asimilarity_search_with_relevance_scores(query, k=k, **kwargs)
        )
        keyword_task = asyncio.create_task(
            self._keyword_search(query, k)
        )
        vec_res, key_res = await asyncio.gather(vector_task, keyword_task)

        # Đưa vector results vào SearchResult objects
        results_map: Dict[str, SearchResult] = {}
        for doc, v_score in vec_res:
            doc_id = doc.metadata.get('id', id(doc))
            results_map[doc_id] = SearchResult(
                document=doc,
                vector_score=v_score,
            )

        # Kết hợp keyword và priority
        for sr in key_res:
            doc_id = sr.document.metadata.get('id', id(sr.document))
            if doc_id in results_map:
                results_map[doc_id].keyword_score = sr.keyword_score
                results_map[doc_id].priority_score = sr.priority_score
            else:
                results_map[doc_id] = sr

        # Tính final score và lọc
        final_list = []
        for sr in results_map.values():
            final_score = (
                self.alpha * sr.vector_score
                + self.beta * sr.keyword_score
                + self.gamma * sr.priority_score
            )
            if final_score >= self.min_final_threshold:
                final_list.append((final_score, sr.document))

        # Sắp xếp và trả top-k
        final_list.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in final_list[:k]]
    
    async def acoolchat_similarity_search(
        self,
        query: str,
        k: int = 3,
        **kwargs
    ) -> List[Document]:
        """Wrapper để gọi hybrid_search"""
        cache_key = f"{query}|{k}"
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]

        documents = await self.hybrid_search(query, k, **kwargs)
        print(f"=== Kết quả hybrid search: {query} ===")
        for i, doc in enumerate(documents, start=1):
            print(f"[{i}] Document:")
            print(doc.page_content)  # in nội dung
            print("-" * 40)
        self.result_cache[cache_key] = documents
        return documents

    @override
    def as_retriever(self, search_kwargs: dict = None) -> BaseRetriever:
        """
        Return instance of CustomCoolChatRetriever, ensure that its interface similar to RetrieverLike
        """
        return CoolChatRetriever(vectorstore=self, search_kwargs=search_kwargs)


class CoolChatRetriever(BaseRetriever):
    """Optimized retriever with caching and batch processing"""
    
    vectorstore: CoolChatVectorStore
    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    cache: TTLCache = Field(default_factory=lambda: TTLCache(maxsize=100, ttl=3600))
    use_hybrid: bool = Field(default=True)  # New parameter
    request_semaphore: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(5)  # <-- annotate
    
    @override
    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.vectorstore.coolchat_similarity_search(query, **self.search_kwargs)
    
    @override
    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        cache_key = f"async_{query}"
        if (docs := self.cache.get(cache_key)) is not None:
            return docs
        async with self.request_semaphore:
            docs = await self.vectorstore.acoolchat_similarity_search(
                query, k = 3, **self.search_kwargs
            )
        self.cache[cache_key] = docs
        return docs
    
    @override
    async def batch_get_relevant_documents(
        self,
        queries: list[str],
        batch_size: int = 5
    ) -> list[list[Document]]:
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            batch_results = await asyncio.gather(*[
                self._aget_relevant_documents(q) for q in batch
            ])
            results.extend(batch_results)
        return results
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
