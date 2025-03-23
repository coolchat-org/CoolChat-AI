import asyncio
from datetime import datetime
import json
import os
from typing import Any, AsyncGenerator, List, Optional, Dict, ClassVar, override
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
from app.models.chatModel import ChatModel
from app.models.ranker import CoolChatVectorStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
import aiohttp  # HTTP async request
from langchain.schema.runnable import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.summarizer import tokenize_and_summarize_openai


NOTIFICATION_ENDPOINT = os.environ.get(
    "NOTIFICATION_ENDPOINT", 
    "https://webhook.site/aec3b4e7-5611-45d5-b267-515d0f486a90"  # Endpoint for testing with end_conversation tool.
)

#####################################################
###  CACHING
###  MODEL, RETRIEVER, PROMT That can be reused
#####################################################
@lru_cache(maxsize=1)
def get_model():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY, streaming=True)

@lru_cache(maxsize=1)
def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_prompts():
    # Pull from langchain hub
    rephrase_prompt = pull("langchain-ai/chat-langchain-rephrase")
    
    return rephrase_prompt

def get_retriever(index_host: str):
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=index_host)
    vectorStore = CoolChatVectorStore(index=index, embedding=get_embedder())
    return vectorStore.as_retriever(search_kwargs={})


def get_history_aware_retriever(index_host: str):
    retriever = get_retriever(index_host)
    rephrase_prompt = get_prompts()
    return create_history_aware_retriever(
        llm=get_model(),
        retriever=retriever,
        prompt=rephrase_prompt
    )

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_ending_conversation = False
        
    @override
    async def on_llm_new_token(self, token: str, **kwargs):
        """Run on new LLM token. Only available when streaming is enabled."""
        await self.queue.put(token)
        
    @override
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Run when tool starts running."""
        # Detect when tool end_conversation is called
        if serialized.get("name") == "end_conversation":
            self.is_ending_conversation = True
            print("End conversation tool started")
    
    @override
    async def on_tool_end(self, output: str, **kwargs):
        """Run when tool ends running."""
        # Mark an end for conversation => use in response_from_LLM
        if self.is_ending_conversation:
            await self.queue.put("[END_SIGNAL]")  # Set special signal

def create_end_conversation_tool(session_id: str):
    """Create a tool for ending the conversation."""
    
    def end_conversation(input_text: str = "") -> str:
        """
        Use this tool when the user wants to end the conversation or say goodbye.
        This should be used when the user says things like "goodbye", "bye", "see you later",
        "thanks for your help, bye", "that's all I needed", etc.
        """

        return "Cảm ơn quý khách đã trò chuyện. Hẹn gặp lại quý khách trong thời gian sớm nhất!"
    
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

class AsyncRAGTool(BaseTool):
    """Tool for performing RAG with async support."""
    
    name: ClassVar[str] = "answer_question"
    description: ClassVar[str] = "Use this to answer user questions by retrieving relevant information"
    
    def __init__(self, db_host: str, chatbot_attitude: str):
        super().__init__()
        self._db_host = db_host
        self._chatbot_attitude = chatbot_attitude

    def _run(self, query: str, chat_history=[]) -> str:
        """Synchronous run method - required but not used."""
        raise NotImplementedError("This tool only supports async execution")
    
    async def _arun(
        self, 
        query: str, 
        chat_history=[], 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Asynchronous run method that performs RAG."""
        # Lấy callbacks từ run_manager nếu có
        callbacks = run_manager.get_child().handlers if run_manager else None
        
        retriever = get_history_aware_retriever(index_host=self._db_host)
        
        question_answer_chain = create_stuff_documents_chain(
            llm=get_model().with_config({"callbacks": callbacks}) if callbacks else get_model(),
            prompt=ChatPromptTemplate.from_messages([
                ("system", f"""Trả lời câu hỏi của khách hàng dựa trên thông tin từ cơ sở tri thức sau:
                
                {{context}}

                Hướng dẫn:
                - Cung cấp câu trả lời DỰA TRÊN THÔNG TIN ĐÃ TRUY XUẤT. Luôn trả lời theo phong cách {self._chatbot_attitude}.
                - Trả lời bằng tiếng Việt (giữ lại các jargon tiếng Anh nếu có).
                - Nếu thông tin không đủ để trả lời câu hỏi, hãy thông báo lịch sự rằng thông tin đó không có sẵn. Bạn không được self-generated câu trả lời để tránh gây hiểu nhầm.
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
        )
        
        rag_chain = create_retrieval_chain(
            retriever=retriever, 
            combine_docs_chain=question_answer_chain
        )
        
        result = await rag_chain.ainvoke(
            {"input": query, "chat_history": chat_history},
            config={"callbacks": callbacks}
        )
        return result["answer"]

def create_rag_tool(db_host: str, chatbot_attitude: str):
    """Create a tool that performs RAG for answering questions."""
    return AsyncRAGTool(db_host=db_host, chatbot_attitude=chatbot_attitude)

def create_unified_agent(llm, db_host: str, streaming_handler=None, session_id: str = None, company_name: str = "CoolChat Consulting Company", chatbot_attitude: str = "professional"):
    """Create a unified agent that can detect goodbyes and perform RAG."""
    
    end_conversation_tool = create_end_conversation_tool(session_id)
    rag_tool = create_rag_tool(db_host, chatbot_attitude)

    system_prompt = f"""Bạn là một tư vấn viên làm việc tại bộ phận dịch vụ khách hàng của {company_name}.

    Bạn có quyền truy cập vào 2 tools:
    1. `answer_question`: Sử dụng công cụ này để trả lời câu hỏi của khách hàng.
    2. `end_conversation`: CHỈ sử dụng công cụ này khi khách hàng rõ ràng muốn kết thúc cuộc hội thoại.

    HÀNH VI TRONG CUỘC HỘI THOẠI:
    - Khi khách hàng chào mở đầu (e.g. Hello, chào bạn): Hãy chào họ một cách lịch sự và giới thiệu ngắn gọn về {company_name}.
    - Luôn luôn sử dụng tool `answer_question` để trả lời các thắc mắc của người dùng.
    - Kết thúc mỗi câu trả lời bằng việc hỏi khách hàng xem còn câu hỏi nào khác không.

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

    # Sử dụng streaming_llm nếu có streaming_handler
    if streaming_handler:
        streaming_llm = llm.with_config({"callbacks": [streaming_handler]})
        agent = create_openai_tools_agent(streaming_llm, [end_conversation_tool, rag_tool], prompt)
    else:
        agent = create_openai_tools_agent(llm, [end_conversation_tool, rag_tool], prompt)

    return AgentExecutor(
        agent=agent,
        tools=[end_conversation_tool, rag_tool],
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[streaming_handler] if streaming_handler else None,
        tool_kwargs={"callbacks": [streaming_handler]} if streaming_handler else {}  # Thêm callbacks vào công cụ
    )

async def response_from_LLM(session_id: str, query: str, db_host: str, company_name: str = "CoolChat Consulting Company", chatbot_attitude: str = "professional"):
    model = get_model()
    
    # Get conversation for history
    conversation = await ChatModel.get(document_id=session_id)
    langchain_history = ChatModel.convert_to_langchain_history(conversation.memory)
    
    # Tạo streaming handler
    streaming_handler = StreamingCallbackHandler()
    
    # Create the unified agent với streaming handler và các tham số tùy chỉnh
    unified_agent = create_unified_agent(
        model, 
        db_host, 
        streaming_handler,
        session_id=session_id,
        company_name=company_name, 
        chatbot_attitude=chatbot_attitude
    )
    
    # Bắt đầu agent execution trong một task riêng biệt
    config = RunnableConfig(callbacks=[streaming_handler])
    agent_task = asyncio.create_task(
        unified_agent.ainvoke(
            {"input": query, "chat_history": langchain_history},
            config=config
        )
    )
    
    accumulated_response = ""
    is_ending = False
    save_task_started = False
    
    async def save_task(is_end: bool):
        """Task lưu vào database chạy bất đồng bộ."""
        if accumulated_response:
            await save_chat_history_and_memory(
                chat=conversation, 
                query=query, 
                response=accumulated_response, 
                old_history=langchain_history,
                is_end=is_end
            )
    
    # Stream từ queue của streaming handler
    try:
        while True:
            try:
                # Lấy token từ queue với timeout
                token = await asyncio.wait_for(streaming_handler.queue.get(), timeout=0.1)
                
                # Phát hiện signal kết thúc đặc biệt
                if token == "[END_SIGNAL]":
                    is_ending = True
                    print("End conversation detected via callback")
                    
                    # Gọi hàm notify_conversation_end tại đây
                    asyncio.create_task(notify_conversation_end(session_id))
                    continue  # Bỏ qua signal này, không stream đến client
                
                # Token thông thường
                accumulated_response += token
                yield token
                
            except asyncio.TimeoutError:
                # Kiểm tra xem agent đã hoàn thành chưa
                if agent_task.done():
                    break
    
    except asyncio.CancelledError:
        # Xử lý khi request bị hủy
        agent_task.cancel()
        raise
    
    finally:
        # Đảm bảo lưu lại cuộc trò chuyện
        if not save_task_started:
            save_task_started = True
            asyncio.create_task(save_task(is_end=is_ending))
        
        if is_ending:
            print("Conversation ended by user")
            is_ending = False

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