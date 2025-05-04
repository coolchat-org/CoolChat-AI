from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, Path, Body
from fastapi.responses import StreamingResponse
from app.api.v2.dependencies import IndexPreparation, prepare_index
from app.services.v2.rag.ragCore import stream_response
from app.services.v2.compares.chatService import parallel_response_from_LLM
from app.services.v2.index.indexService import createIndexesFromFilesAndUrls, disposeVirtualIndex, saveVirtualIndex
from app.dtos.chatUserDto import ChatUserDto
from app.dtos.indexDto import CreateNamespaceDto, SaveNamespaceDto
from app.models.chatModel import ChatModel


v2_router = APIRouter(prefix="/v2/chat", tags=["chat-v2-vps"])

    
@v2_router.post("/training-chatbot/{org_id}", response_model=CreateNamespaceDto, summary="Create a vector database to train the chatbot",
    responses={
        400: {"description": "Invalid file format or metadata structure"},
        401: {"description": "Unauthorized - Invalid/Missing API Key"},
        413: {"description": "Request entity too large - File size limit exceeded"},
        500: {"description": "Internal server error during index creation"}
    }
)
async def create_knowledge_database(
    org_id: str = Path(..., description="Organization ID"),
    prep: IndexPreparation = Depends(prepare_index)
    ):
    """
    This endpoint hashes the organization ID to generate a unique database name and processes the uploaded files and URLs to build an index.

    **Parameters:**
    - **org_id**: *str*  
        The unique organization ID provided as a path parameter (e.g., "org123").
    - **files**: *List[UploadFile]*  
        A list of uploaded files via form data. This includes PDFs, DOCX, TXT files, and a mandatory `metadata.json`.

    **metadata.json:**
    
    The `metadata.json` file should follow the structure below:

    ```json
    {
        "files": {
            "example.pdf": 1,
            "example.docx": 2
        },
        "urls": [
            {"url": "https://example.com", "priority": 1}
        ]
    }
    ```

    **Returns:**
      - **CreateNamespaceDto**: An object containing information about the created index, such as success status and a message.
    """
    try:
        result = await createIndexesFromFilesAndUrls(
            prep.urls,
            prep.pdf_files, prep.docx_files, prep.txt_files,
            prep.pdf_prio, prep.docx_prio, prep.txt_prio,
            organization_id=org_id
        )
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@v2_router.post("/create-virtual-chatbot/{org_id}", response_model=CreateNamespaceDto, summary="Create a virtual chatbot", description=
    """
        Compare chatbots (training virtual chatbot). It will modify the organization id 
        (e.g. abcd -> abcd_0) and use the operation's result to create a virtual database name. The parameters are the same as /training-chatbot.
    """, 
    responses={
        400: {"description": "Invalid file format or metadata structure"},
        401: {"description": "Unauthorized - Invalid/Missing API Key"},
        413: {"description": "Request entity too large - File size limit exceeded"},
        500: {"description": "Internal server error during index creation"}
})
async def create_virtual_chatbot(
    org_id: str = Path(..., description="Organization ID"),
    prep: IndexPreparation = Depends(prepare_index)
):
    try:
        result = await createIndexesFromFilesAndUrls(
            prep.urls,
            prep.pdf_files, prep.docx_files, prep.txt_files,
            prep.pdf_prio, prep.docx_prio, prep.txt_prio,
            organization_id=org_id,
            is_virtual=True
        )
        result.message = "Virtual Index created!"
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    

@v2_router.post("/save-virtual-chatbot/{org_id}", response_model=CreateNamespaceDto, summary="Save the virtual chatbot", responses={
        401: {"description": "Unauthorized - Invalid/Missing API Key"},
        500: {"description": "Internal server error during saving virtual chatbot"}
})
async def save_virtual_chatbot(org_id: str = Path(..., description="Organization ID"),
user_data: SaveNamespaceDto = Body(..., description="Current namespace of Organization")) -> Any:
    """
    Save the virtual chatbot by removing the previous index.

    **Parameters**:
    - **org_id**: *str* - Organization ID (required).
    - **user_data**: *SaveNamespaceDto* - Current namespace of Organization

    **Returns**:
    - The result of the operations (including new namespace for virtual bot!)

    **Raises**:
    - HTTPException: If an error occurs during saving
    """
    try:
        result = saveVirtualIndex(old_namespace=user_data.current_namespace, organization_id=org_id)
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
@v2_router.delete("/dispose-virtual-chatbot/{org_id}", response_model=CreateNamespaceDto, summary="Delete the virtual chatbot", responses={
        401: {"description": "Unauthorized - Invalid/Missing API Key"},
        500: {"description": "Internal server error during disposing virtual chatbot"}
})
async def dispose_virtual_chatbot(org_id: str = Path(..., description="Organization ID"),
user_data: SaveNamespaceDto = Body(..., description="Current namespace of Organization")) -> Any:
    """
    Remove the virtual chatbot index.

    **Parameters:**
    - **org_id**: *str*  
        The unique organization ID provided as a path parameter (e.g., "org123").
    - **user_data**: *SaveNamespaceDto*  
        The current namespace of the organization, provided in the request body.

    **Returns:**
    - **CreateNamespaceDto**: An object containing the result of the disposal operation.

    **Raises:**
    - **HTTPException**: If an error occurs during the disposal process.
    """
    try:
        result = disposeVirtualIndex(old_namespace=user_data.current_namespace, organization_id=org_id)
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@v2_router.put("/new-chat", response_model=Any)
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


@v2_router.get("/{id}", response_model=Any)
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

@v2_router.post("/{id}", response_model=Any)
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
            stream_response(
                session_id=id,
                query=user_data.new_message,
                db_host=user_data.index_host,
                namespace=user_data.namespace,
                company_name=user_data.config.company_name,
                chatbot_attitude=user_data.config.chatbot_attitude,
                start_sentence=user_data.config.start_sentence,
                end_sentence=user_data.config.end_sentence
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


@v2_router.post("/compare/{org_id}", response_model=Any)
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
                db_host=user_data.index_host,
                namespace_1=user_data.namespace,
                namespace_2=user_data.virtual_namespace,
                company_name=user_data.config.company_name,
                chatbot_attitude=user_data.config.chatbot_attitude,
                start_sentence=user_data.config.start_sentence,
                end_sentence=user_data.config.end_sentence
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
    