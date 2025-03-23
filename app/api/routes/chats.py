import json
from typing import Any, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Path, Body
from fastapi.responses import StreamingResponse
from app.api.services.chatService import parallel_response_from_LLM
from app.api.services.agentService import response_from_LLM
from app.api.services.indexService import createIndexesFromFilesAndUrls, disposeVirtualIndex, saveVirtualIndex
from app.dtos.chatUserDto import ChatUserDto, CreateIndexDto, SaveIndexDto
from app.models.chatModel import ChatModel


agent_router = APIRouter(prefix="/chat", tags=["chat"])

#####################################################
###  MANIPULATING 
###  BOT'S KNOWLEDGE
#####################################################
@agent_router.post("/training-chatbot/{org_id}", response_model=CreateIndexDto)
async def create_knowledge_database(org_id: str = Path(..., description="Organization ID"),
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
                elif file.content_type == "text/plain" or file.content_type == "text/plain; charset=utf-8":
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

@agent_router.post("/create-virtual-chatbot/{org_id}", response_model=CreateIndexDto)
async def create_virtual_chatbot(org_id: str = Path(..., description="Organization ID"),
    # handling passing files from frontend
    files: List[UploadFile] = File(..., description="Files to create index from")
    ) -> Any:
    """
    Compare chatbots (training virtual chatbot). It will modify the organization id (e.g. abcd -> abcd_0) and use the operation's result to create a virtual database name.

    @params

    Root:
    - org_id: organization id

    Form-Data:
    - files: list of uploaded files, including metadata file

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
                elif file.content_type == "text/plain" or file.content_type == "text/plain; charset=utf-8":
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
        
        result = await createIndexesFromFilesAndUrls(url_prio_list, pdfFiles, docxFiles, txtFiles, pdfPrio, docxPrio, txtPrio, organization_id=org_id, is_virtual=True)

        result.message = "Virtual Index created!"

        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
@agent_router.post("/save-virtual-chatbot/{org_id}", response_model=CreateIndexDto)
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
    
@agent_router.delete("/dispose-virtual-chatbot/{org_id}", response_model=CreateIndexDto)
def dispose_virtual_chatbot(org_id: str = Path(..., description="Chat ID"),
user_data: SaveIndexDto = Body(..., description="User's Input Query and Chat history")) -> Any:
    """
    Removing the virtual index.

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

@agent_router.put("/new-chat", response_model=Any)
async def create_new_chat() -> Any:
    """
    Create new chat. Use its _id returned field to operate in other routes.

    @params

    Empty

    @returns

    - info of the new chat.
    """
    new_conversation = await ChatModel.init_session()

    return new_conversation


@agent_router.get("/{id}", response_model=Any)
async def read_items(id: str = Path(..., description="Chat ID")) -> Any:
    """
    Get all info of a specific conversation.

    @params

    Root:
    - id: chat id

    @returns

    - info of a chat.
    """
    conversation = await ChatModel.get(id)
    return conversation

@agent_router.post("/{id}", response_model=Any)
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
    - virtual_db_host: not fill, null
    - config: {
        company_name: string,
        chatbot_attitude: string
    }

    @returns

    - a streaming response 
    """
    try:
        return StreamingResponse(
            response_from_LLM(
                session_id=id,
                query=user_data.new_message,
                db_host=user_data.org_db_host,
                company_name=user_data.config.company_name,
                chatbot_attitude=user_data.config.chatbot_attitude
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


@agent_router.post("/compare/{org_id}", response_model=Any)
async def create_parallel_reply_message(
    org_id: str = Path(..., description="Organization ID"),
    user_data: ChatUserDto = Body(..., description="User's Input Query and Chat history")
) -> Any:
    """
    Get two reply messages from the LLM.

    @params

    Root:
    - org_id: Organization ID

    Body:
    - new_message: string
    - org_db_host: url string
    - virtual_db_host: url string
    - config: {
        company_name: string,
        chatbot_attitude: string
    }

    @returns
    {
        "response_1": result, 
        "response_2": None, 
        "done": False,
        "full_response_1": "".join(full_response_1),
        "full_response_2": "".join(full_response_2)
    }
    At the end of streaming, we do not contain the response_1, response_2 field.

    """
    try:
        async def response_generator():
            async for chunk in parallel_response_from_LLM(
                chat_id=id,
                query=user_data.new_message,
                db_host_1=user_data.org_db_host,
                db_host_2=user_data.virtual_db_host,
                company_name=user_data.config.company_name,
                chatbot_attitude=user_data.config.chatbot_attitude
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
    