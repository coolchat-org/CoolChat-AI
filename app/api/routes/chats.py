import json
from typing import Any, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Path, Body
from fastapi.responses import StreamingResponse
from app.api.services.chatService import parallel_response_from_LLM, response_from_LLM
from app.api.services.graphService import create_CHATBOT_graph
from app.api.services.indexService import createIndexesFromFilesAndUrls, disposeVirtualIndex, saveVirtualIndex
from app.dtos.chatUserDto import ChatUserDto, CreateIndexDto, LLMResponseDto, SaveIndexDto
from app.models.chatModel import ChatModel
from app.core.db import mongo_connection


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/training-chatbot/{org_id}", response_model=CreateIndexDto)
async def create_knowledge_database(org_id: str = Path(..., description="Chat ID"),
    # handling passing files from frontend
    files: List[UploadFile] = File(..., description="Files to create index from and metadata.json"),
    ) -> Any:
    """
    Create vector database (training chatbot). It will hash the organization id and use the operation's result to create a unique database name.

    @params

    Root:
    - org_id: organization id

    Form-Data:
    - files: list of uploaded files, including metadata file.

    @returns

    - Info of the index, which will be send to backend.
    """
    try:
        metadata_content = None

        if files:
            pdfFiles, docxFiles, txtFiles = [], [], []
            pdfPrio, docxPrio, txtPrio = [], [], []
            metafile_idx = None

            for idx, file in enumerate(files):
                if file.filename == "metadata.json":
                    metadata_content = await file.read()
                    metadata_content = json.loads(metadata_content.decode("utf-8"))
                    print("Metadata:", metadata_content)
                    metafile_idx = idx
                """
                metadata.json:
                {
                    "files": {
                        filename: priority1, 
                        filename2: priority2
                    },
                    "urls": [
                        {"url": string, "priority": 1},
                        {"url": string, "priority": 1} 
                    ]
                }
                """
            del files[metafile_idx]    
            
            for file in files:
                if file.content_type == "application/pdf":
                    pdfFiles.append(file)
                    pdfPrio.append(metadata_content["files"][file.filename])
                elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    docxFiles.append(file)
                    docxPrio.append(metadata_content["files"][file.filename])
                elif file.content_type == "text/plain":
                    txtFiles.append(file)
                    txtPrio.append(metadata_content["files"][file.filename])
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported file type: {file.content_type}. Only pdf, docx, doc, txt accepted.",
                    )

        url_prio_list = []
        if metadata_content is not None:
            if metadata_content["urls"] is not None:
                url_prio_list = metadata_content["urls"]  

        result = await createIndexesFromFilesAndUrls(url_prio_list, pdfFiles, docxFiles, txtFiles, pdfPrio, docxPrio, txtPrio, organization_id=org_id)

        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

## The following routes are used to manipulate with training chatbots
@router.post("/create-virtual-chatbot/{org_id}", response_model=CreateIndexDto)
async def create_virtual_chatbot(org_id: str = Path(..., description="Chat ID"),
    # handling passing files from frontend
    files: List[UploadFile] = File(..., description="Files to create index from"),
    urls: str = Form(..., description="List of URLs as JSON string")
    ) -> Any:
    """
    Compare chatbots (training virtual chatbot). It will modify the organization id (e.g. abcd -> abcd_0) and use the operation's result to create a virtual database name.

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
                elif file.content_type in [
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    docFiles.append(file)
                elif file.content_type == "text/plain":
                    txtFiles.append(file)

        web_url: str = "https://vnexpress.net/than-thanh-hoa-ielts-4627600.html?utm_source=facebook&utm_medium=fanpage_VnE&utm_campaign=phuonguyen&fbclid=IwAR342qeRaOfJRwnJUl145GW3ojO2-S2XGcHa1XvxSqMTc6mKplJE2siI_qE"

        url_list = urls.split(",") if urls else [web_url]
        
        result = await createIndexesFromFilesAndUrls(url_list, pdfFiles, docFiles, txtFiles, organization_id=org_id, is_virtual=True)

        result.message = "Virtual Index created!"

        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
@router.post("/save-virtual-chatbot/{org_id}", response_model=CreateIndexDto)
def save_virtual_chatbot(org_id: str = Path(..., description="Chat ID"),
user_data: SaveIndexDto = Body(..., description="User's Input Query and Chat history")) -> Any:
    """
    Save the virtual chatbot by removing the previous index.

    @params

    Root:
    - org_id: organization id

    @returns

    - The result of the operations.
    """
    try:
        
        result = saveVirtualIndex(old_index_name=user_data.old_index_name, organization_id=org_id)
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
@router.delete("/dispose-virtual-chatbot/{org_id}", response_model=CreateIndexDto)
def dispose_virtual_chatbot(org_id: str = Path(..., description="Chat ID"),
user_data: SaveIndexDto = Body(..., description="User's Input Query and Chat history")) -> Any:
    """
    Save the virtual chatbot by removing the previous index.

    @params

    Root:
    - org_id: organization id

    @returns

    - The result of the operations.
    """
    try:
        
        result = disposeVirtualIndex(old_index_name=user_data.old_index_name, organization_id=org_id)
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

# DEPRECATED
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


# New version: Langgraph integration
# @router.post("/{id}", response_model=Any)
# async def create_reply_message(
#     id: str = Path(..., description="Chat ID"),
#     user_data: ChatUserDto = Body(..., description="User's Input Query and Chat history")
# ) -> Any:
#     """
#     Get Reply message from the LLM.

#     @params

#     Root:
#     - id: chat id

#     Body:
#     - new_message: string
#     - org_db_host: url string

#     @returns

#     - a streaming response 
#     """
#     try:
#         graph = create_CHATBOT_graph()

#         # Init state
#         initial_state = {
#             "messages": [], # Previous message
#             "current_message": user_data.new_message,
#             "should_create_transaction": False,
#             "chat_id": id,
#             "db_host": user_data.org_db_host
#         }

#         async def response_generator():
#             # Execute graph and get final state
#             final_state = await graph.ainvoke(initial_state)
#             # Get the last AI message
#             response = final_state["message"][-1].content
#             # Stream the response
#             for chunk in response.split():
#                 yield f"data: {chunk}\n\n"
            

#         return StreamingResponse(
#             response_generator(), 
#             media_type="text/event_stream"
#         )

#     except HTTPException as http_exc:
#         raise http_exc

#     except Exception as e:
#         # Handle unexpected errors
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"An unexpected error occurred: {str(e)}"
#         )

@router.post("/compare/{id}", response_model=Any)
async def create_parallel_reply_message(
    id: str = Path(..., description="Chat ID"),
    user_data: ChatUserDto = Body(..., description="User's Input Query and Chat history")
) -> Any:
    """
    Get two reply messages from the LLM.
    """
    try:
        async def response_generator():
            async for chunk in parallel_response_from_LLM(
                chat_id=id,
                query=user_data.new_message,
                db_host_1=user_data.org_db_host,
                db_host_2=user_data.virtual_db_host 
            ):
                # Format as SSE event
                yield f"data: {chunk}\n\n"

        return StreamingResponse(
            response_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )