import asyncio
from typing import Any, List
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.hub import pull
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.core.connections import PineconeConnectionPool
from app.models.chatModel import ChatModel
from app.models.rankest import CoolChatVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.summarizer import tokenize_and_summarize_openai
from cachetools import TTLCache, cached
from app.services.v2.rag.constant import *


# Cache instances
_model_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_embedder_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_prompt_cache = TTLCache(maxsize=PROMPT_CACHE_SIZE, ttl=CACHE_TTL)
_retriever_cache = TTLCache(maxsize=RETRIEVER_CACHE_SIZE, ttl=CACHE_TTL)
_history_retriever_cache = TTLCache(maxsize=HISTORY_RETRIEVER_CACHE_SIZE, ttl=CACHE_TTL)
_question_answer_chain_cache = TTLCache(maxsize=QA_CACHE_SIZE, ttl=CACHE_TTL)

@lru_cache(maxsize=1)
def get_prompts():
    return pull("langchain-ai/chat-langchain-rephrase")

# Dynamic components - dùng TTLCache
@cached(cache=_model_cache)
def get_model():
    # return ChatOpenAI(
    #     model="gpt-4o-mini", 
    #     temperature=0.7, 
    #     api_key=settings.OPENAI_API_KEY, 
    #     streaming=True
    # )
    # decrease the temperature to reduce inference time
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", 
        api_key=settings.GOOGLE_AI_API_KEY,
        temperature=0.5, disable_streaming=False)

@cached(cache=_embedder_cache)
def get_embedder():
    return OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=settings.OPENAI_API_KEY
    )

@cached(cache=_retriever_cache)
def get_retriever_cached(index_host: str, namespace: str):
    return get_retriever(index_host, namespace)

@cached(cache=_history_retriever_cache)
def get_history_aware_retriever_cached(index_host: str, namespace: str):
    return get_history_aware_retriever(index_host, namespace)

@cached(cache=_prompt_cache)
def get_cached_qa_prompt(company: str, start_with: str, end_with: str, style: str) -> ChatPromptTemplate:
    return get_qa_prompt_template(company, start_with, end_with, style)

def get_retriever(index_host: str, namespace: str):
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

@cached(cache=_question_answer_chain_cache)
def get_question_answer_chain(company: str, start_with: str, end_with: str, style: str):
    llm = get_model()
    prompt_template = get_cached_qa_prompt(company, start_with, end_with, style)
    return create_stuff_documents_chain(llm=llm, prompt=prompt_template)

def _get_style_example(style: str) -> str:
    examples = {
        "friendly": "Dạ, mình xin giải đáp...", 
        "formal": "Kính thưa quý khách, theo thông tin từ hệ thống...",
        "professional": "Chúng tôi xin trả lời câu hỏi của bạn...",
        "funny": "Dạ vâng mình sẽ nói về ..."
    }
    return examples.get(style, "")

def get_qa_prompt_template(company: str, start_with: str, end_with: str, style: str) -> ChatPromptTemplate:
    """Tạo prompt template dựa trên chatbot attitude"""
    return ChatPromptTemplate.from_messages([
        ("system", f"""
        Trả lời câu hỏi của khách hàng dựa trên thông tin từ cơ sở tri thức sau bằng Tiếng Việt (giữ lại các jargon tiếng Anh).

        Hướng dẫn:
        - Cung cấp câu trả lời DỰA TRÊN CONTEXT ĐÃ TRUY XUẤT, trả lời theo phong cách {style}. Tuy nhiên để tránh sự không tự nhiên, đừng nói rằng Dựa trên thông tin được cung cấp hay tri thức có sẵn, mà hãy giải đáp tự nhiên vào.
        - Nếu trong câu hỏi có kèm lời chào, nhớ chào theo mẫu '{start_with}' kèm giới thiệu bạn là trợ lý ảo đến từ công ty '{company}'
        - Nếu thông tin không đủ để trả lời câu hỏi, hãy thông báo lịch sự -> 'Hiện tại tôi chưa thể hỗ trợ thông tin này cho bạn, bạn có thể đặt câu hỏi khác hoặc liên hệ nhân viên để được tư vấn trực tiếp. Rất xin lỗi vì sự bất tiện này'. Bạn không được self-generated câu trả lời để tránh gây hiểu nhầm.
        - Câu hỏi lạc chủ đề: 
           VD: Người dùng hỏi "Cách nấu phở?" hay những chuyện trên trời dưới đất -> "Tôi chỉ hỗ trợ câu hỏi về {company}. Bạn muốn hỏi gì về [dịch vụ/sản phẩm] của chúng tôi không?"
        - Nếu người dùng tạm biệt, hãy tạm biệt họ theo mẫu câu '{end_with}'

        Context của bạn: {{context}}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

async def stream_response(session_id: str, query: str, db_host: str, namespace: str, company_name: str = DEFAULT_COMPANY, chatbot_attitude: str = DEFAULT_ATTITUDE, start_sentence: str = DEFAULT_START_MSG, end_sentence: str = DEFAULT_END_MSG):
    # Lấy lịch sử chat
    conversation = await ChatModel.get(document_id=session_id)
    if conversation.memory is None:
        history = []
    else:
        history = ChatModel.convert_to_langchain_history(conversation.memory)
    
    # Khởi tạo retriever
    retriever = get_retriever(db_host, namespace)
    # llm = get_model()
    # prompt_template = get_cached_qa_prompt(company_name, start_sentence, end_sentence, chatbot_attitude)
        
    question_answer_chain = get_question_answer_chain(company_name, start_sentence, end_sentence, chatbot_attitude)
    
    rag_chain = create_retrieval_chain(
        retriever=retriever, 
        combine_docs_chain=question_answer_chain
    )
    
    full_resp = []
    async for chunk in rag_chain.astream({"input": query, "chat_history": history}):
        if answer := chunk.get("answer"):
            full_resp.append(answer)
            yield answer

    # Khi loop kết thúc, lưu history
    asyncio.create_task(save_chat_history_and_memory(
        chat=conversation,
        query=query,
        response="".join(full_resp),
        old_history=history,
    ))


async def save_chat_history_and_memory(chat: ChatModel, query: str, response: str, old_history: List[AIMessage | HumanMessage], is_end: bool = False):
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