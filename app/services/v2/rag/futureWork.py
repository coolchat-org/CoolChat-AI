import asyncio
from datetime import datetime
import json
import os
from typing import Any, List, Optional, Dict, ClassVar, override
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.hub import pull
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from app.core.config import settings
from app.core.connections import PineconeConnectionPool
from app.models.chatModel import ChatModel
# from app.models.ranker import CoolChatVectorStore
from app.models.ranker_new import CoolChatVectorStore

from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
import aiohttp  # HTTP async request
from langchain.schema.runnable import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.summarizer import tokenize_and_summarize_openai
from cachetools import TTLCache, cached

NOTIFICATION_ENDPOINT = os.environ.get(
    "NOTIFICATION_ENDPOINT", 
    "https://webhook.site/aec3b4e7-5611-45d5-b267-515d0f486a90"  # Endpoint for testing with end_conversation tool.
)

# Constants
CACHE_TTL = 3600  # 1 hour
DEFAULT_COMPANY = "CoolChat Consulting Company"
DEFAULT_ATTITUDE = "professional"
DEFAULT_START_MSG = "Chào bạn, đây là ban tư vấn khách hàng của công ty chúng tôi"
DEFAULT_END_MSG = "Cảm ơn quý khách đã trò chuyện. Hẹn gặp lại quý khách trong thời gian sớm nhất!"

# Cache configurations
RETRIEVER_CACHE_SIZE = 100
PROMPT_CACHE_SIZE = 10
AGENT_CACHE_SIZE = 100

# Cache instances
_model_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_embedder_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_prompt_cache = TTLCache(maxsize=PROMPT_CACHE_SIZE, ttl=CACHE_TTL)
_retriever_cache = TTLCache(maxsize=RETRIEVER_CACHE_SIZE, ttl=CACHE_TTL)
_history_aware_retriever_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
_unified_agent_cache = TTLCache(maxsize=AGENT_CACHE_SIZE, ttl=CACHE_TTL)

# Static prompt - giữ lru_cache
@lru_cache(maxsize=1)
def get_prompts():
    return pull("langchain-ai/chat-langchain-rephrase")

# Dynamic components - dùng TTLCache
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
    return get_retriever(index_host, namespace)

@cached(cache=_history_aware_retriever_cache)
def get_history_aware_retriever_cached(index_host: str, namespace: str):
    return get_history_aware_retriever(index_host, namespace)

@cached(cache=_prompt_cache)
def get_cached_qa_prompt(chatbot_attitude: str) -> ChatPromptTemplate:
    return get_qa_prompt_template(chatbot_attitude)

@cached(cache=_unified_agent_cache)
def get_cached_unified_agent(
    db_host: str, 
    namespace: str, 
    company_name: str,
    chatbot_attitude: str,
    start_sentence: str,
    end_sentence: str
):
    model = get_model()
    return create_unified_agent(
        model, db_host, 
        namespace=namespace,
        company_name=company_name,
        chatbot_attitude=chatbot_attitude,
        start_sentence=start_sentence,
        end_sentence=end_sentence
    )

def get_retriever(index_host: str, namespace: str):
    # pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    # index = pc.Index(host=index_host)
    pool = PineconeConnectionPool.get_instance()
    index = pool.get_index(index_host)
    vectorStore = CoolChatVectorStore(index=index, embedding=get_embedder())
    return vectorStore.as_retriever(search_kwargs={"namespace": namespace})


def get_history_aware_retriever(index_host: str, namespace: str):
    retriever = get_retriever_cached(index_host, namespace)
    rephrase_prompt = get_prompts()
    return create_history_aware_retriever(
        llm=get_model(),
        retriever=retriever,
        prompt=rephrase_prompt
    )


def create_end_conversation_tool(end_sentence: str):
    """Create a tool for ending the conversation."""
    
    def end_conversation(input_text: str = "") -> str:
        """
        Use this tool when the user wants to end the conversation or say goodbye.
        This should be used when the user says things like "goodbye", "bye", "see you later",
        "thanks for your help, bye", "that's all I needed", etc.
        """

        return end_sentence
    
    return Tool.from_function(
        func=end_conversation,
        name="end_conversation",
        description="Use this when the user is saying goodbye or wants to end the conversation"
    )

async def notify_conversation_end(session_id: str):
    """Gửi thông báo đến backend rằng cuộc trò chuyện đã kết thúc."""
    try:
        # Tạo payload với thông tin chi tiết để dễ debug
        payload = {
            "conv_id": session_id,
            "is_end": True,
        }
        
        # Log thông tin ra console để debug
        print(f"Sending end notification for session {session_id} to {NOTIFICATION_ENDPOINT}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Gửi request đến endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(NOTIFICATION_ENDPOINT, json=payload) as response:
                response_text = await response.text()
                status = response.status
                
                # Log kết quả
                print(f"Notification result: Status={status}")
                print(f"Response: {response_text[:200]}...")  # Hiển thị 200 ký tự đầu của response
                
                # Lưu vào file log để xem lại sau (tuỳ chọn)
                with open("notification_log.txt", "a") as f:
                    f.write(f"[{datetime.now().isoformat()}] Session: {session_id}, Status: {status}\n")
                    
                return status == 200

    except Exception as e:
        print(f"Error in notification: {str(e)}")
        # Lưu lỗi vào file log (tuỳ chọn)
        with open("notification_error_log.txt", "a") as f:
            f.write(f"[{datetime.now().isoformat()}] Error for session {session_id}: {str(e)}\n")
        return False

def get_qa_prompt_template(chatbot_attitude: str) -> ChatPromptTemplate:
    """Tạo prompt template dựa trên chatbot attitude"""
    return ChatPromptTemplate.from_messages([
        ("system", f"""Trả lời câu hỏi của khách hàng dựa trên thông tin từ cơ sở tri thức sau bằng Tiếng Việt (giữ lại các jargon tiếng Anh nếu có):
        
        {{context}}

        Hướng dẫn:
        - Cung cấp câu trả lời DỰA TRÊN THÔNG TIN ĐÃ TRUY XUẤT. Luôn trả lời theo phong cách {chatbot_attitude}.
        - Nếu thông tin không đủ để trả lời câu hỏi, hãy thông báo lịch sự rằng thông tin đó không có sẵn. Bạn không được self-generated câu trả lời để tránh gây hiểu nhầm.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


class AsyncRAGTool(BaseTool):
    """Tool for performing RAG with async support."""
    
    name: ClassVar[str] = "answer_question"
    description: ClassVar[str] = "Use this to answer user questions by retrieving relevant information"
    
    def __init__(self, db_host: str, namespace: str, chatbot_attitude: str):
        super().__init__()
        self._db_host = db_host
        self._namespace = namespace
        self._chatbot_attitude = chatbot_attitude

    def _run(self, query: str, chat_history=[]) -> str:
        """Synchronous run method - required but not used."""
        raise NotImplementedError("This tool only supports async execution")
    
    async def _arun(
        self, 
        query: str, 
        chat_history=[], 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ):
        """Asynchronous run method that performs RAG."""
        retriever = get_history_aware_retriever_cached(
            index_host=self._db_host, 
            namespace=self._namespace
        )
        # Get cached prompt template
        prompt_template = get_cached_qa_prompt(self._chatbot_attitude)

        # Create model instance with callbacks if needed
        llm = get_model()
        
        # Create chain - không cache vì phụ thuộc callbacks
        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt_template
        )
        
        rag_chain = create_retrieval_chain(
            retriever=retriever, 
            combine_docs_chain=question_answer_chain
        )
        
        result = await rag_chain.ainvoke(
            {"input": query, "chat_history": chat_history}
        )
        return result["answer"]

def create_rag_tool(db_host: str, namespace: str, chatbot_attitude: str):
    """Create a tool that performs RAG for answering questions."""
    return AsyncRAGTool(db_host=db_host, namespace=namespace, chatbot_attitude=chatbot_attitude)

def create_unified_agent(llm, db_host: str, streaming_handler=None, namespace: str = None, company_name: str = "CoolChat Consulting Company", chatbot_attitude: str = "professional", start_sentence: str = "Chào bạn, đây là ban tư vấn khách hàng của công ty chúng tôi", end_sentence: str = "Cảm ơn quý khách đã trò chuyện. Hẹn gặp lại quý khách trong thời gian sớm nhất!"):
    """Create a unified agent that can detect goodbyes and perform RAG."""
    
    end_conversation_tool = create_end_conversation_tool(end_sentence=end_sentence)
    rag_tool = create_rag_tool(db_host, namespace, chatbot_attitude)

    system_prompt = f"""Bạn là một tư vấn viên làm việc tại bộ phận dịch vụ khách hàng của {company_name}.

    Bạn có quyền truy cập vào 2 tools:
    1. `answer_question`: LUÔN Sử dụng công cụ này để trả lời câu hỏi của khách hàng.
    2. `end_conversation`: CHỈ sử dụng công cụ này khi khách hàng rõ ràng muốn kết thúc cuộc hội thoại.

    HÀNH VI TRONG CUỘC HỘI THOẠI:
    - Khi khách hàng chào mở đầu (e.g. Hello, chào bạn): Hãy chào họ một cách lịch sự với câu mẫu sau: "{start_sentence}" và giới thiệu ngắn gọn về {company_name}.
    - Luôn luôn sử dụng tool `answer_question` để trả lời các thắc mắc của người dùng, và chỉ gọi nó một lần cho mỗi câu hỏi.
    - Sau khi nhận được kết quả từ công cụ `answer_question`, hãy trả về kết quả đó và không tiếp tục xử lý thêm. Nếu dữ liệu trả về có nhắc đến {company_name}, có thể paraphrase là 'chúng tôi', 'công ty chúng tôi'..etc để tránh lặp tên doanh nghiệp và tổ chức.
    - Kết thúc mỗi câu trả lời bằng việc hỏi khách hàng xem còn câu hỏi nào khác không.
    - Trả lời bằng tiếng Việt

    KẾT THÚC CUỘC HỘI THOẠI:
    - Khi khách hàng nói lời chào tạm biệt: "goodbye", "bye", "thank you and goodbye", "tạm biệt", etc.: Use the `end_conversation` tool.
    - Return EXACTLY the output from the tool without modification.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, [end_conversation_tool, rag_tool], prompt)

    return AgentExecutor(
        agent=agent,
        tools=[end_conversation_tool, rag_tool],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=1,
    )


async def response_from_LLM(session_id: str, query: str, 
            db_host: str, namespace: str,
            company_name: str = "CoolChat Consulting Company", chatbot_attitude: str = "professional", start_sentence: str = "Chào bạn, đây là ban tư vấn khách hàng của công ty chúng tôi",
            end_sentence: str = "Cảm ơn quý khách đã trò chuyện. Hẹn gặp lại quý khách trong thời gian sớm nhất!"
):
    model = get_model()
    
    # Lấy lịch sử cuộc trò chuyện
    conversation = await ChatModel.get(document_id=session_id)
    langchain_history = ChatModel.convert_to_langchain_history(conversation.memory)
    
    # Tạo agent thống nhất (không cần streaming_handler)
    unified_agent = get_cached_unified_agent(
        db_host, namespace, company_name, 
        chatbot_attitude, start_sentence, end_sentence
    )
    
    # Sử dụng hàng đợi để streaming
    streaming_queue = asyncio.Queue()
    is_ending = False
    
    async def stream_agent_response():
        end_signal_received = False
        async for event in unified_agent.astream_events({"input": query, "chat_history": langchain_history}, version='v2'):
            if event["event"] == "on_tool_start" and event["name"] == "end_conversation":
                await streaming_queue.put("[END_SIGNAL]")
                end_signal_received = True
            elif event["event"] == "on_tool_end" and event["name"] == "end_conversation":
                # Lấy câu trả lời từ công cụ và gửi vào hàng đợi
                tool_output = event["data"]["output"]
                await streaming_queue.put(tool_output)
            elif event["event"] == "on_chat_model_stream" and not end_signal_received:
                token = event["data"]["chunk"].content
                await streaming_queue.put(token)
            elif end_signal_received:
                break
        
    # Bắt đầu task streaming
    agent_task = asyncio.create_task(stream_agent_response())
    
    accumulated_response = ""
    save_task_started = False
    
    # Streaming từ hàng đợi
    while True:
        try:
            token = await asyncio.wait_for(streaming_queue.get(), timeout=0.1)
            if token == "[END_SIGNAL]":
                is_ending = True
                continue
            accumulated_response += token
            yield token
        except asyncio.TimeoutError:
            if agent_task.done():
                break
    
    # Lưu lịch sử cuộc trò chuyện sau khi streaming
    if not save_task_started:
        save_task_started = True
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

async def save_chat_history_and_memory(chat: ChatModel, query: str, response: str, old_history: List[AIMessage | HumanMessage], is_end: bool):
    # print("User: ", query)
    # print("AI: ", response)
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