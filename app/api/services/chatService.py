import asyncio
from typing import Any, List, Tuple
# from fastapi import HTTPException, status
from langchain.hub import pull
# from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
import openai
from app.core.config import settings
from app.models.chatModel import ChatModel
from app.core.db import mongo_connection
from pinecone import Pinecone
from functools import lru_cache
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Tải model/tokenizer tại cấp module
MODEL_PATH = "CreatorPhan/ViSummary"
tokenizer = None
model = None


@lru_cache(maxsize=1)
def get_model():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY, streaming=True)

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

def get_retriever(index_host: str):
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=index_host)
    vectorStore = PineconeVectorStore(index=index, embedding=get_embedder())
    return vectorStore.as_retriever(search_kwargs={'k': 2})  # Giảm k xuống 1

def get_history_aware_retriever(index_host: str):
    retriever = get_retriever(index_host)
    _, rephrase_prompt = get_prompts()
    return create_history_aware_retriever(
        llm=get_model(),
        retriever=retriever,
        prompt=rephrase_prompt
    )

def create_LC_history(pair: Any):
    if isinstance(pair, str):
        return [HumanMessage(content=pair)]
    user_msg = HumanMessage(content=pair[0])
    ai_msg = AIMessage(content=pair[1])

    return [user_msg, ai_msg]

async def response_from_LLM(chat_id: str, query: str, db_host: str):
    question_answer_chain = get_chain()
    history_aware_retriever = get_history_aware_retriever(index_host=db_host)

    ragChain = create_retrieval_chain(
        retriever=history_aware_retriever, 
        combine_docs_chain=question_answer_chain
    )

    conversation = await ChatModel.find_by_id(chat_id=chat_id, collection=mongo_connection.db["chats"])
    if conversation.memory == "":
        chat_history = []
    else:
        chat_history = conversation.memory
    
    accumulated_response = ""
    async def save_task():
        """Task lưu vào database chạy bất đồng bộ."""
        if accumulated_response:
            await save_chat_history_and_memory(chat=conversation, query=query, response=accumulated_response, old_history=langchain_history)


    save_task_started = False  
    langchain_history: List[AIMessage | HumanMessage] = []

    messages = list(map(create_LC_history, chat_history))  
    if len(chat_history) != 0:
        langchain_history.extend([msg for pair in messages for msg in pair])  # Flatten

    async for chunk in ragChain.astream(input={"input": query, "chat_history": langchain_history}):
        if chunk.get('answer', None) is not None:
            answer = chunk['answer']
            accumulated_response += answer  # Accumulate response
            yield answer  # Stream to client

    # Save after streaming
    if not save_task_started:
        save_task_started = True
        asyncio.create_task(save_task())


async def save_chat_history_and_memory(chat: ChatModel, query: str, response: str, old_history: List[AIMessage | HumanMessage]):
    print("User: ", query)
    print("AI: ", response)
    print(f"Context: {old_history}")
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
        chat.memory = [new_context]
    else:
        chat.memory.append([query, response])
    
    updated_chat = await chat.save(mongo_connection.db["chats"])
    return updated_chat

def load_model_and_tokenizer():
    """
    Tải mô hình và tokenizer chỉ khi cần thiết.
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Load summarization model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to('cpu')
    return tokenizer, model

def tokenize_and_summarize(old_context, query, response, max_token: int = 10024) -> Tuple[str, bool]:
    new_context = f"{old_context + " " if old_context != "" else ""}Customer: {query}, AI: {response}"
    token_count = len(new_context.split())
    
    if token_count >= max_token:
        tokenizer, model = load_model_and_tokenizer()
        tokens = tokenizer(f"Tóm tắt văn bản sau: {new_context}", return_tensors='pt').input_ids
        output = model.generate(tokens.to('cpu'), max_new_tokens=max_token // 2)[0]
        predict = tokenizer.decode(output, skip_special_tokens=True)

        return predict, True

    return "", False

def tokenize_and_summarize_openai(old_context, query, response, max_token: int = 2048) -> Tuple[str, bool]:
    new_context = f"{old_context + " " if old_context != "" else ""}Customer: {query}, AI: {response}"
    token_count = len(new_context.split())
    
    openai.api_key = settings.OPENAI_API_KEY
    if token_count >= max_token:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu cần chất lượng cao hơn

                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý AI chuyên tóm tắt văn bản và cuộc hội thoại bằng tiếng Việt."},
                    {"role": "user", "content": f"Tóm tắt văn bản sau, là một cuộc hội thoại giữa một nhân viên chăm sóc khách hàng và khách hàng: {new_context}"}
                ],
                max_tokens=max_token // 2,
                temperature=0.7,  # Điều chỉnh độ sáng tạo (0.7 là giá trị cân bằng)
                n=1,  # Số lượng kết quả trả về
                stop=None,  # Không cần chuỗi dừng cụ thể
            )
            # Trích xuất kết quả tóm tắt
            summary = response.choices[0].message.content.strip()
            return summary, True
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "", False

    return "", False