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
from app.core import db
from app.core.config import settings
from app.dtos.chatUserDto import LLMResponseDto
from app.models.chatModel import ChatModel
from app.models.models import PyObjectId

# BaseMessage[] to work with chat history
chat_history: List[BaseMessage] = []

def response_from_LLM(index_path: str, query: str):
    retrieval_qa_chat_prompt: ChatPromptTemplate = pull("davidnguyen2212/retrievala-qa-chat-for-cool-chat")
    rephrase_prompt: ChatPromptTemplate = pull("langchain-ai/chat-langchain-rephrase")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.OPENAI_API_KEY)
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=settings.OPENAI_API_KEY)
    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=retrieval_qa_chat_prompt)

    vectorStore: FAISS = FAISS.load_local(folder_path=index_path, embeddings=embedder, allow_dangerous_deserialization=True)
    retriever: VectorStoreRetriever = vectorStore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm=model, retriever=retriever, prompt=rephrase_prompt
    )

    ragChain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=question_answer_chain)
    result = ragChain.invoke(input={"input": query, "chat_history": chat_history})
    if result:
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(result["answer"]))

    return LLMResponseDto(
        reply=result["answer"],
        context=result["context"],
        message="Response successfully"
    )

async def updateConversation(chatId: PyObjectId, userData: str, llmData: str):
    chat = await ChatModel.find_by_id(chatId, db["chats"])
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This conversation ID: {chatId} does not exist"
        )
    chat.conversations.append([userData, llmData])
    updated_Chat = await chat.save(db["chats"])