from typing import Any, Dict, List, Tuple, override
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStore
from pydantic import Field

class CoolChatVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", **kwargs):
        super().__init__(index=index, embedding=embedding, text_key=text_key, **kwargs)
        self.w_s = 0.6
        self.w_p = 0.4

    def coolchat_similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        """
        The customized function to search on our CoolChat vector database.
        It follows our document-reranking metrics:

        - w_p, w_s are the weight of document's priority and score respectively
        - priority and score are scaled to [0, 1]
        - Weighted sum: w_p * priority + w_s * score

        Finally, choose the biggest two weighted sums and return the documents corresponding to them.
        """
        results: List[Tuple[Document, float]] = self.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            **kwargs
        )

        # score already in [0, 1]. We must scale the priority in this range.
        priorities: List[int] = [
            int(doc_and_score[0].metadata.get("priority", "1")) 
            for doc_and_score in results
        ] 
        max_prio, min_prio = max(priorities), min(priorities)
        if max_prio == min_prio:
            scaled_priorities = [1 for _ in range(len(priorities))]
        else:
            scaled_priorities = [(x - min_prio) / (max_prio - min_prio) for x in priorities]

        # Arrange!
        sorted_results = sorted(
            enumerate(results), 
            key=lambda elem: scaled_priorities[elem[0]] * self.w_p + elem[1][1] * self.w_s, 
            reverse=True
        )

        sorted_results = [elem[1][0] for elem in sorted_results]
        print("Contexts: ")
        print(sorted_results)
        # Delete duplication by "id"
        unique_docs = {doc.metadata.get("id", id(doc)): doc for doc in sorted_results}.values()
        return list(unique_docs)
    
    async def acoolchat_similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        results: List[Tuple[Document, float]] = await self.asimilarity_search_with_relevance_scores(query, k=k, **kwargs)
  
        priorities: List[int] = [
            int(doc_and_score[0].metadata.get("priority", "1")) 
            for doc_and_score in results
        ] 
        max_prio, min_prio = max(priorities), min(priorities)
        if max_prio == min_prio:
            scaled_priorities = [1 for _ in range(len(priorities))]
        else:
            scaled_priorities = [(x - min_prio) / (max_prio - min_prio) for x in priorities]

        # Arrange!
        sorted_results = sorted(
            enumerate(results), 
            key=lambda elem: scaled_priorities[elem[0]] * self.w_p + elem[1][1] * self.w_s, 
            reverse=True
        )

        # Shortener version!
        sorted_results = [elem[1][0] for elem in sorted_results]
        sorted_results = sorted_results[:2]
        # Delete duplication by "id"
        unique_docs = {doc.metadata.get("id", id(doc)): doc for doc in sorted_results}.values()
        return list(unique_docs)
    
    @override
    def as_retriever(self, search_kwargs: dict = None) -> BaseRetriever:
        """
        Return instance of CustomCoolChatRetriever, ensure that its interface similar to RetrieverLike
        """
        return CoolChatRetriever(vectorstore=self, search_kwargs=search_kwargs)

class CoolChatRetriever(BaseRetriever):
    vectorstore: CoolChatVectorStore
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    @override
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.coolchat_similarity_search(query, **self.search_kwargs)
    
    @override
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return await self.vectorstore.acoolchat_similarity_search(query, **self.search_kwargs)
    
    class Config:
        arbitrary_types_allowed = True