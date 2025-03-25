import asyncio
import json
from typing import Any, List
# from fastapi import HTTPException, status
from langchain.hub import pull
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from app.core.config import settings
from app.models.chatModel import ChatModel
from pinecone import Pinecone
from functools import lru_cache

from app.models.ranker import CoolChatVectorStore
from app.utils.summarizer import tokenize_and_summarize_openai


@lru_cache(maxsize=1)
def get_model():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY, streaming=True)

@lru_cache(maxsize=1)
def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_prompts(company_name: str, chatbot_attitude: str):
    system_prompt = f"""Bạn là một tư vấn viên làm việc tại bộ phận dịch vụ khách hàng của {company_name}.

    HÀNH VI TRONG CUỘC HỘI THOẠI:
    - Khi khách hàng chào mở đầu (e.g. Hello, chào bạn): Hãy chào họ một cách lịch sự bằng tiếng Việt và giới thiệu ngắn gọn về {company_name}.
    - Nếu bạn nhận thấy câu hỏi đã xuất hiện trong lịch sử chat (không phải lời chào tạm biệt), hãy trả lời câu hỏi đó một cách nhanh chóng. Nếu không, với các câu hỏi hoặc thông tin nhập vào thông thường (không phải lời chào), hãy cung cấp câu trả lời DỰA TRÊN THÔNG TIN ĐÃ TRUY XUẤT. Luôn trả lời theo phong cách {chatbot_attitude} và bằng tiếng Việt!
    - Khi thông tin không đủ: Thông báo lịch sự với khách hàng rằng thông tin đó không có sẵn.
    - Kết thúc mỗi câu trả lời bằng việc hỏi khách hàng xem còn câu hỏi nào khác không.
    - Nếu khách hàng đặt câu hỏi bằng tiếng Anh: hãy trả lời bằng tiếng Anh, nếu không thì trả lời bằng tiếng Việt như bình thường (giữ lại các jargon tiếng Anh nếu có).
    
    KẾT THÚC CUỘC HỘI THOẠI:
    - Khi khách hàng nói lời chào tạm biệt, “bye”, “thank you and goodbye”, v.v.: hãy chào tạm biệt khách hàng theo phong cách {chatbot_attitude}!
    
    Không bao giờ tự tạo câu trả lời mà không sử dụng hệ thống truy xuất thông tin để tránh gây hiểu nhầm! Chỉ sử dụng thông tin có sẵn trong 
    
    {{context}}.
    """

    retrieval_qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return (
        retrieval_qa_prompt,
        pull("langchain-ai/chat-langchain-rephrase"),
    )

@lru_cache(maxsize=1)
def get_chain(company_name: str, chatbot_attitude: str):
    retrieval_qa_chat_prompt, _ = get_prompts(company_name, chatbot_attitude)
    model = get_model()
    return create_stuff_documents_chain(llm=model, prompt=retrieval_qa_chat_prompt)

def get_retriever(index_host: str):
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=index_host)
    vectorStore = CoolChatVectorStore(index=index, embedding=get_embedder())
    return vectorStore.as_retriever(search_kwargs={})


def get_history_aware_retriever(index_host: str, company_name: str, chatbot_attitude: str):
    retriever = get_retriever(index_host)
    _, rephrase_prompt = get_prompts(company_name, chatbot_attitude)
    return create_history_aware_retriever(
        llm=get_model(),
        retriever=retriever,
        prompt=rephrase_prompt
    )


async def parallel_response_from_LLM(chat_id: str, query: str, db_host_1: str, db_host_2: str, company_name: str, chatbot_attitude: str):
    question_answer_chain = get_chain(company_name, chatbot_attitude)
    history_aware_retriever_1 = get_history_aware_retriever(db_host_1, company_name, chatbot_attitude)
    history_aware_retriever_2 = get_history_aware_retriever(db_host_2, company_name, chatbot_attitude)

    # Tạo hai pipeline
    ragChain1 = create_retrieval_chain(retriever=history_aware_retriever_1, combine_docs_chain=question_answer_chain)
    ragChain2 = create_retrieval_chain(retriever=history_aware_retriever_2, combine_docs_chain=question_answer_chain)

    async def stream_rag_chain(rag_chain):
        async for chunk in rag_chain.astream({"input": query, "chat_history": []}):
            if chunk.get('answer', None) is not None:
                answer = chunk['answer']
                yield answer  # Stream to client

    # Create aiterator 
    stream1 = stream_rag_chain(ragChain1)
    stream2 = stream_rag_chain(ragChain2)
    task1 = asyncio.create_task(stream1.__anext__())
    task2 = asyncio.create_task(stream2.__anext__())

    # Add accumulators for both responses
    full_response_1 = []
    full_response_2 = []

    while True:
        tasks = {t for t in [task1, task2] if t is not None}
        if not tasks:
            # Send final accumulated responses
            yield json.dumps({
                "done": True,
                "full_response_1": "".join(full_response_1),
                "full_response_2": "".join(full_response_2)
            })
            return
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for completed_task in done:
            try:
                result = await completed_task
                if completed_task == task1:
                    full_response_1.append(result)
                    response_data = {
                        "response_1": result, 
                        "response_2": None, 
                        "done": False,
                        "full_response_1": "".join(full_response_1),
                        "full_response_2": "".join(full_response_2)
                    }
                    yield json.dumps(response_data)
                    task1 = asyncio.create_task(stream1.__anext__())
                else:
                    full_response_2.append(result)
                    response_data = {
                        "response_1": None, 
                        "response_2": result, 
                        "done": False,
                        "full_response_1": "".join(full_response_1),
                        "full_response_2": "".join(full_response_2)
                    }
                    yield json.dumps(response_data)
                    task2 = asyncio.create_task(stream2.__anext__())
            except StopAsyncIteration:
                if completed_task == task1:
                    task1 = None
                else:
                    task2 = None

