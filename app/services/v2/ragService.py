import asyncio
import json
from datetime import datetime
from functools import lru_cache
import os
from typing import List, Optional
import aiohttp
from cachetools import TTLCache, cached
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.models.ranker_new import CoolChatVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from app.core.config import settings
from app.core.connections import PineconeConnectionPool
from app.models.chatModel import ChatModel
from langchain.hub import pull
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.summarizer import tokenize_and_summarize_openai

# Environment variables
NOTIFICATION_ENDPOINT = os.environ.get(
    "NOTIFICATION_ENDPOINT", 
    "https://webhook.site/aec3b4e7-5611-45d5-b267-515d0f486a90"
)

# Constants
CACHE_TTL = 3600  # 1 hour
DEFAULT_COMPANY = "CoolChat Consulting Company"
DEFAULT_ATTITUDE = "professional"
DEFAULT_START_MSG = "Chào bạn, đây là ban tư vấn khách hàng của công ty chúng tôi"
DEFAULT_END_MSG = "Cảm ơn quý khách đã trò chuyện. Hẹn gặp lại quý khách trong thời gian sớm nhất!"

# Cache instances
_model_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_embedder_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_retriever_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
_history_aware_retriever_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)

# Cache functions
@cached(cache=_model_cache)
def get_model():
    return ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.7, 
        api_key=settings.OPENAI_API_KEY, 
        streaming=True
    )

@cached(cache=_embedder_cache)
def get_embedder():
    return OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=settings.OPENAI_API_KEY
    )

@cached(cache=_retriever_cache)
def get_retriever_cached(index_host: str, namespace: str):
    pool = PineconeConnectionPool.get_instance()
    index = pool.get_index(index_host)
    vectorStore = CoolChatVectorStore(index=index, embedding=get_embedder())
    return vectorStore.as_retriever(search_kwargs={"namespace": namespace})

@cached(cache=_history_aware_retriever_cache)
def get_history_aware_retriever_cached(index_host: str, namespace: str):
    retriever = get_retriever_cached(index_host, namespace)
    rephrase_prompt = lru_cache(maxsize=1)(lambda: pull("langchain-ai/chat-langchain-rephrase"))()
    return create_history_aware_retriever(
        llm=get_model(),
        retriever=retriever,
        prompt=rephrase_prompt
    )

def get_combined_prompt_template(company_name: str, chatbot_attitude: str, start_sentence: str, end_sentence: str):
    return ChatPromptTemplate.from_messages([
        ("system", f"""Bạn là một tư vấn viên làm việc tại bộ phận dịch vụ khách hàng của {company_name}.

Hướng dẫn:
- Đầu tiên, nếu khách hàng chào bạn, hãy chào lại họ lịch sự với mẫu câu: {start_sentence}
- Nếu khách hàng nói lời chào tạm biệt như "goodbye", "bye", "tạm biệt", etc., hãy trả lời với "{end_sentence}"
- Ngược lại, hãy trả lời câu hỏi của khách hàng dựa trên thông tin từ cơ sở tri thức sau:
{{context}}

Lưu ý:
- Luôn trả lời theo phong cách {chatbot_attitude}.
- Trả lời bằng tiếng Việt.
- Nếu thông tin không đủ, hãy thông báo lịch sự.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

async def response_from_LLM(session_id: str, query: str, db_host: str, namespace: str, 
                            company_name: str = DEFAULT_COMPANY, 
                            chatbot_attitude: str = DEFAULT_ATTITUDE, 
                            start_sentence: str = DEFAULT_START_MSG,
                            end_sentence: str = DEFAULT_END_MSG):
    # Lấy lịch sử cuộc trò chuyện
    conversation = await ChatModel.get(document_id=session_id)
    langchain_history = ChatModel.convert_to_langchain_history(conversation.memory)
    
    # Lấy model và retriever từ cache
    model = get_model()
    retriever = get_history_aware_retriever_cached(db_host, namespace)
    
    # Tạo prompt tích hợp
    prompt = get_combined_prompt_template(company_name, chatbot_attitude, start_sentence, end_sentence)
    
    # Tạo document chain với streaming
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    
    # Tạo retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Streaming
    accumulated_response = ""
    is_ending = False
    async for event in retrieval_chain.astream({"input": query, "chat_history": langchain_history}):
        yield event["answer"]
        # # 3) yield it and accumulate
        # accumulated_response += token
        # yield token

        # # 4) break on your end token
        # if token.strip() == "[END]":
        #     break
        
        # Lưu lịch sử cuộc trò chuyện
    asyncio.create_task(save_chat_history_and_memory(
        chat=conversation, 
        query=query, 
        response=accumulated_response, 
        old_history=langchain_history,
        is_end=is_ending
    ))

# Thông báo kết thúc cuộc trò chuyện nếu cần
    if is_ending:
        asyncio.create_task(notify_conversation_end(session_id))

async def notify_conversation_end(session_id: str):
    """Gửi thông báo đến backend rằng cuộc trò chuyện đã kết thúc."""
    try:
        payload = {"conv_id": session_id, "is_end": True}
        async with aiohttp.ClientSession() as session:
            async with session.post(NOTIFICATION_ENDPOINT, json=payload) as response:
                status = response.status
                return status == 200
    except Exception as e:
        print(f"Error in notification: {str(e)}")
        return False

async def save_chat_history_and_memory(chat: ChatModel, query: str, response: str, old_history: List[AIMessage | HumanMessage], is_end: bool):
    print("User: ", query)
    print("AI: ", response)
    chat.conversations.append([query, response])
    if len(old_history) != 0:
        old_context: str = " ".join([msg.content for msg in old_history]) 
        new_context, context_summarized = tokenize_and_summarize_openai(
            old_context, query, response, max_token=2048
        )
    else:
        old_context = ""
        context_summarized = False
    
    if context_summarized:
        # chat.memory = [new_context]
        chat.memory = [{ "type": "human", "content": new_context }]
    else:
        chat.memory += [
                {
                    "type": "human",
                    "content": query,
                }, 
                {
                    "type": "ai",
                    "content": response,
                }
            ]
    
    if is_end:
        updated_chat = await chat.close_session()
    else:
        updated_chat = await chat.save_session()
    return updated_chat