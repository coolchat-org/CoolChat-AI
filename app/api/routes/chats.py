from typing import Any, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Path, Body
from fastapi.responses import StreamingResponse
from app.api.services.chatService import response_from_LLM
from app.api.services.indexService import createIndexesFromFilesAndUrls
from app.dtos.chatUserDto import ChatUserDto, CreateIndexDto, LLMResponseDto
from app.models.chatModel import ChatModel
from app.core.db import mongo_connection


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/training-chatbot/{org_id}", response_model=CreateIndexDto)
async def create_knowledge_database(org_id: str = Path(..., description="Chat ID"),
    # handling passing files from frontend
    files: List[UploadFile] = File(..., description="Files to create index from"),
    urls: str = Form(..., description="List of URLs as JSON string")
    ) -> Any:
    """
    Create vector database (training chatbot). It will hash the organization id and use the operation's result to create a unique database name.

    @params

    Root:
    - org_id: organization id

    Form-Data:
    - files: list of uploaded files
    - urls: url list (string representation), separated by comma (comma only). Eg: "https://example1.com,https://example2.com"

    @returns

    - Info of the index, which will be send to backend.
    """
    try:
        if files:
            pdfFiles, docFiles, txtFiles = [], [], []
            for file in files:
                if file.content_type not in ["application/pdf", "application/msword", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported file type: {file.content_type}. Only pdf, docx, doc, txt accepted.",
                    )
                elif file.content_type == "application/pdf":
                    pdfFiles.append(file)
                elif file.content_type == "text/plain":
                    docFiles.append(file)
                else:
                    txtFiles.append(file)

        web_url: str = "https://vnexpress.net/than-thanh-hoa-ielts-4627600.html?utm_source=facebook&utm_medium=fanpage_VnE&utm_campaign=phuonguyen&fbclid=IwAR342qeRaOfJRwnJUl145GW3ojO2-S2XGcHa1XvxSqMTc6mKplJE2siI_qE"

        url_list = urls.split(",") if urls else [web_url]
        
        result = await createIndexesFromFilesAndUrls(url_list, pdfFiles, docFiles, txtFiles, organization_id=org_id)
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.put("/new-chat", response_model=Any)
async def create_new_chat() -> Any:
    """
    Create new chat. Use its _id returned field to operate in other routes.

    @params

    Empty

    @returns

    - info of the new chat.
    """
    new_conversation = await ChatModel.init(collection=mongo_connection.db["chats"])

    return new_conversation


@router.get("/{id}", response_model=Any)
async def read_items(id: str = Path(..., description="Chat ID")) -> Any:
    """
    Get all info of a specific conversation.

    @params

    Root:
    - id: chat id

    @returns

    - info of a chat.
    """
    conversation = await ChatModel.find_by_id(chat_id=id, collection=mongo_connection.db["chats"])
    return conversation


@router.post("/{id}", response_model=Any)
async def create_reply_message(
    id: str = Path(..., description="Chat ID"),
    user_data: ChatUserDto = Body(..., description="User's Input Query and Chat history")
) -> Any:
    """
    Get Reply message from the LLM.

    @params

    Root:
    - id: chat id

    Body:
    - new_message: string
    - org_db_host: url string

    @returns

    - a streaming response
    """
    try:
        return StreamingResponse(
            response_from_LLM(
                chat_id=id,
                query=user_data.new_message,
                db_host=user_data.org_db_host
            ), 
            media_type="text/event_stream"
        )

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
