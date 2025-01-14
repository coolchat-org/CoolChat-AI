from typing import List
from click import prompt
from fastapi import HTTPException, status
from langchain.hub import pull
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_pinecone import PineconeVectorStore
from app.core import db
from app.core.config import settings
from app.dtos.chatUserDto import LLMResponseDto
from app.models.chatModel import ChatModel
from app.models.models import PyObjectId
from pinecone import Pinecone
from functools import lru_cache

# BaseMessage[] to work with chat history
chat_history: List[BaseMessage] = []


@lru_cache(maxsize=1)
def get_model():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.OPENAI_API_KEY, streaming=True)

@lru_cache(maxsize=1)
def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_prompts():
    return (
        pull("davidnguyen2212/retrievala-qa-chat-for-cool-chat"),
        pull("langchain-ai/chat-langchain-rephrase"),
    )

@lru_cache(maxsize=1)
def get_chain():
    retrieval_qa_chat_prompt, _ = get_prompts()
    model = get_model()
    return create_stuff_documents_chain(llm=model, prompt=retrieval_qa_chat_prompt)

@lru_cache(maxsize=1)
def get_retriever():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host="org-index-7812b41e26a080f5-oei5fcs.svc.aped-4627-b74a.pinecone.io")
    vectorStore = PineconeVectorStore(index=index, embedding=get_embedder())
    return vectorStore.as_retriever(search_kwargs={'k': 1})  # Giảm k xuống 1

@lru_cache(maxsize=1)
def get_history_aware_retriever():
    retriever = get_retriever()
    _, rephrase_prompt = get_prompts()
    return create_history_aware_retriever(
        llm=get_model(),
        retriever=retriever,
        prompt=rephrase_prompt
    )



async def response_from_LLM(query: str, index_path: str = None):
    question_answer_chain = get_chain()
    history_aware_retriever = get_history_aware_retriever()

    ragChain = create_retrieval_chain(
        retriever=history_aware_retriever, 
        combine_docs_chain=question_answer_chain
        )
    # result = ragChain.invoke(input={"input": query, "chat_history": chat_history})
    # if result:
    #     chat_history.append(HumanMessage(content=query))
    #     chat_history.append(AIMessage(result["answer"]))

    # return LLMResponseDto(
    #     reply=result["answer"],
    #     # context=result["context"],
    #     context=None,
    #     message="Response successfully"
    # )
    async for chunk in ragChain.astream(input={"input": query, "chat_history": chat_history}):
        # content = chunk.replace("\n", "<br>")
        if chunk.get('answer', None) is not None:
            yield f"{chunk['answer']}"

async def updateConversation(chatId: PyObjectId, userData: str, llmData: str):
    chat = await ChatModel.find_by_id(chatId, db["chats"])
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This conversation ID: {chatId} does not exist"
        )
    chat.conversations.append([userData, llmData])
    updated_Chat = await chat.save(db["chats"])