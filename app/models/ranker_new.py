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

@dataclass
class SearchResult:
    document: Document
    vector_score: float
    keyword_score: float = 0.0
    
    @property
    def combined_score(self) -> float:
        return self.vector_score * 0.7 + self.keyword_score * 0.3

class CoolChatVectorStore(PineconeVectorStore):
    """Optimized vector store with caching and efficient search"""
    
    @override
    def __init__(
        self, 
        index, 
        embedding, 
        text_key: str = "text",
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        w_s: float = 0.6,
        w_p: float = 0.4,
        min_similarity: float = 0.5,  # Ngưỡng độ tương đồng tối thiểu
        **kwargs
    ):
        super().__init__(index=index, embedding=embedding, text_key=text_key, **kwargs)
        self.w_s = w_s
        self.w_p = w_p
        self.doc_cache = DocumentCache(maxsize=cache_size, ttl=cache_ttl)
        self._batch_size = 50  # For batch processing
        # Initialize TF-IDF vectorizer for keyword search
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.doc_texts = []  # Store document texts for TF-IDF
        self.doc_mapping = {}  # Map TF-IDF indices to Documents
        self.result_cache = TTLCache(maxsize=2000, ttl=7200)  # 2 hours
        self.embedding_cache = TTLCache(maxsize=5000, ttl=86400)  # 24 hours
        self.min_similarity = min_similarity
        
    def _get_document_id(self, doc: Document) -> str:
        """Get unique identifier for document"""
        return doc.metadata.get("id", str(hash(doc.page_content)))
    
    def _deduplicate_docs(self, docs: List[Document], seen: Optional[Set[str]] = None) -> List[Document]:
        """Efficient document deduplication"""
        if seen is None:
            seen = set()
            
        unique_docs = []
        for doc in docs:
            doc_id = self._get_document_id(doc)
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    async def _process_batch(
        self, 
        docs_scores: List[Tuple[Document, float]]
    ) -> List[Document]:
        """Process a batch of documents efficiently"""
        # Extract priorities and cache documents
        priorities = []
        for doc, _ in docs_scores:
            doc_id = self._get_document_id(doc)
            self.doc_cache.set(doc_id, doc)
            priorities.append(int(doc.metadata.get("priority", "1")))
        
        # Scale priorities efficiently using cached function
        scaled_priorities = PriorityScaler.scale_priorities(tuple(priorities))
        
        # Calculate weighted scores
        weighted_scores = [
            (doc, scaled_priorities[i] * self.w_p + score * self.w_s)
            for i, (doc, score) in enumerate(docs_scores)
        ]
        
        # Sort by weighted score and extract documents
        sorted_docs = [doc for doc, _ in sorted(weighted_scores, key=lambda x: x[1], reverse=True)]
        
        return self._deduplicate_docs(sorted_docs)

    async def _keyword_search(
        self, 
        query: str,
        k: int = 3
    ) -> List[SearchResult]:
        """Perform keyword-based search using TF-IDF"""
        if not self.doc_texts:
            return []
            
        # Preprocess query
        processed_query = self._preprocess_text(query)
        query_vector = self.tfidf.transform([processed_query])
        
        # Calculate similarities
        similarities = (query_vector @ self.tfidf_matrix.T).toarray()[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = self.doc_mapping[idx]
                results.append(SearchResult(
                    document=doc,
                    vector_score=0.0,
                    keyword_score=float(similarities[idx])
                ))
        
        return results

    # @override
    async def acoolchat_similarity_search_original(
        self,
        query: str,
        k: int = 3,
        **kwargs
    ) -> List[Document]:
        """Optimized similarity search"""
        try:
            print(f"=== Kết quả cho query old acoolchat: {query} ===")
            # Check result cache first
            cache_key = f"{query}_{k}"
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]

            # Parallel processing
            embedding = await self._get_cached_embedding(query)
            
            # Batch vector search with reduced k for better performance
            initial_k = min(k * 2, 10)  # Fetch more candidates initially
            results = await asyncio.gather(
                self.index.afetch(embedding, top_k=initial_k, **kwargs),
                self._keyword_search(query, k=initial_k)  # Parallel keyword search
            )
            
            vector_results, keyword_results = results
            
            # Efficient processing of results
            processed_results = await self._process_results_efficiently(
                vector_results, 
                keyword_results,
                k=k
            )
            
            # **Chèn print ở đây**
            print(f"=== Kết quả cho query: {query} ===")
            for i, doc in enumerate(processed_results, start=1):
                print(f"[{i}] Document ID: {self._get_document_id(doc)}")
                print(doc.page_content)  # in nội dung
                print("-" * 40)

            # Cache results
            self.result_cache[cache_key] = processed_results
            return processed_results

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    async def _process_results_efficiently(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int
    ) -> List[Document]:
        """Efficient result processing"""
        # Use sets for faster lookups
        seen_docs = set()
        final_results = []
        
        # Process in parallel if needed
        if len(vector_results) + len(keyword_results) > 100:
            return await self._parallel_process_results(
                vector_results,
                keyword_results,
                k
            )
            
        # Fast processing for smaller result sets
        for doc, score in sorted(
            vector_results + keyword_results,
            key=lambda x: x[1],
            reverse=True
        )[:k]:
            doc_id = self._get_document_id(doc)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                final_results.append(doc)
                
        return final_results[:k]

    async def _get_cached_embedding(self, text: str) -> List[float]:
        """Cache embeddings để tránh tính toán lại"""
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        embedding = await self.embedding.aembed_query(text)
        self.embedding_cache[cache_key] = embedding
        return embedding

    @override
    def coolchat_similarity_search(
        self, 
        query: str, 
        k: int = 3, 
        **kwargs
    ) -> List[Document]:
        """Synchronous version of similarity search"""
        print(f"=== Kết quả cho query: {query} ===")

        results = self.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            **kwargs
        )
        
        # Use asyncio to run async processing in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._process_batch(results))[:k]

    @override
    async def batch_similarity_search(
        self, 
        queries: List[str], 
        k: int = 3, 
        **kwargs
    ) -> List[List[Document]]:
        """Batch process multiple queries efficiently"""
        tasks = [
            self.acoolchat_similarity_search(query, k=k, **kwargs)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for keyword search"""
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\s]', ' ', text)
        # Convert to lowercase and strip
        return text.lower().strip()
    
    def _initialize_keyword_search(self, documents: List[Document]):
        """Initialize keyword search with documents"""
        self.doc_texts = []
        self.doc_mapping = {}
        
        for idx, doc in enumerate(documents):
            processed_text = self._preprocess_text(doc.page_content)
            self.doc_texts.append(processed_text)
            self.doc_mapping[idx] = doc
            
        if self.doc_texts:
            self.tfidf_matrix = self.tfidf.fit_transform(self.doc_texts)
    
    async def _keyword_search(
        self, 
        query: str,
        k: int = 3
    ) -> List[SearchResult]:
        """Perform keyword-based search using TF-IDF"""
        if not self.doc_texts:
            return []
            
        # Preprocess query
        processed_query = self._preprocess_text(query)
        query_vector = self.tfidf.transform([processed_query])
        
        # Calculate similarities
        similarities = (query_vector @ self.tfidf_matrix.T).toarray()[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = self.doc_mapping[idx]
                results.append(SearchResult(
                    document=doc,
                    vector_score=0.0,
                    keyword_score=float(similarities[idx])
                ))
        
        return results

    async def hybrid_search(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.7,  # Weight for vector search
        **kwargs
    ) -> List[Document]:
        """
        Perform hybrid search combining vector and keyword-based search
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword search)
            **kwargs: Additional arguments for vector search
        """
        try:
            # Parallel execution of vector and keyword search
            vector_results, keyword_results = await asyncio.gather(
                self.asimilarity_search_with_relevance_scores(query, k=k, **kwargs),
                self._keyword_search(query, k=k)
            )
            
            # Convert vector results to SearchResult objects
            vector_search_results = [
                SearchResult(document=doc, vector_score=score)
                for doc, score in vector_results
            ]
            
            # Combine results
            all_results: Dict[str, SearchResult] = {}
            
            # Process vector results
            for result in vector_search_results:
                doc_id = self._get_document_id(result.document)
                all_results[doc_id] = result
            
            # Process keyword results
            for result in keyword_results:
                doc_id = self._get_document_id(result.document)
                if doc_id in all_results:
                    # Update existing result with keyword score
                    all_results[doc_id].keyword_score = result.keyword_score
                else:
                    # Add new result
                    all_results[doc_id] = result
            
            # Calculate final scores
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: (
                x.vector_score * alpha + x.keyword_score * (1-alpha)
            ), reverse=True)

            filtered_results = [
                result for result in final_results
                if (result.vector_score * alpha + result.keyword_score * (1 - alpha)) >= self.min_similarity
            ]

            print(f"=== Kết quả hybrid search: {query} ===")
            for i, doc in enumerate(filtered_results[:2], start=1):
                print(f"[{i}] Document:")
                print(doc.document.page_content)  # in nội dung
                print("-" * 40)
            
            # Return top-k documents
            return [result.document for result in filtered_results[:2]]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []

    @override
    async def acoolchat_similarity_search(
        self,
        query: str,
        k: int = 3,
        use_hybrid: bool = True,  # New parameter to control search method
        **kwargs
    ) -> List[Document]:
        """Enhanced similarity search with hybrid option"""
        if use_hybrid:
            return await self.hybrid_search(query, k=k, **kwargs)
        
        
        # Fall back to original vector search if hybrid is disabled
        return await self.acoolchat_similarity_search_original(query, k=k, **kwargs)
    
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
                query, use_hybrid=self.use_hybrid, **self.search_kwargs
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
