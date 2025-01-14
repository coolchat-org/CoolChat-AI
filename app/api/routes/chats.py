import os
from typing import Any
from fastapi import APIRouter, HTTPException, status, Path, Body
from fastapi.responses import StreamingResponse
from app.api.services.chatService import response_from_LLM, updateConversation
from app.api.services.indexService import createIndexesFromFiles
from app.dtos.chatUserDto import ChatUserDto, CreateIndexDto, LLMResponseDto
from app.models.chatModel import ChatModel
from app.core.db import mongo_connection


router = APIRouter(prefix="/chat", tags=["chat"])



@router.post("/create-index", response_model=CreateIndexDto)
async def createLocalIndex() -> Any:
    """
    Create new item.
    """
    try:
        # demo folder path
        relative_folder: str = os.path.join(os.pardir, os.pardir, "docs")  # Lùi 2 cấp vào thư mục "docs"
        exact_folder: str = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_folder))
        # print("Exact folder: ", exact_folder)

        # demo site url
        web_url: str = "https://vnexpress.net/than-thanh-hoa-ielts-4627600.html?utm_source=facebook&utm_medium=fanpage_VnE&utm_campaign=phuonguyen&fbclid=IwAR342qeRaOfJRwnJUl145GW3ojO2-S2XGcHa1XvxSqMTc6mKplJE2siI_qE"
        
        result = await createIndexesFromFiles(exact_folder, [web_url])
        return result

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.put("/new-chat", response_model=str)
def createNewChat() -> Any:
    """
    Update an item.
    """
    
    return "ok"


@router.get("/{id}", response_model=Any)
async def read_items(id: str = Path(..., description="Chat ID")) -> Any:
    """
    Retrieve items.
    """
    conversation = await ChatModel.find_by_id(chat_id=id, collection=mongo_connection.db["chats"])
    return conversation


@router.post("/{id}", response_model=Any)
async def createReplyMsg(
    id: str = Path(..., description="Chat ID"),
    user_data: ChatUserDto = Body(..., description="User Input Query")
) -> Any:
    """
    Get Reply message from the LLM.
    """
    try:
        # print(id)
        # relative_folder: str = os.path.join(os.pardir, os.pardir, "indexes")  # Lùi 2 cấp vào thư mục "docs"
        # exact_folder: str = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_folder))
        # print("Exact folder: ", exact_folder)

        # reply_from_llm = response_from_LLM(query=user_data.new_message)

        # if isinstance(reply_from_llm, Exception):
        #     raise HTTPException(
        #         status.HTTP_400_BAD_REQUEST,
        #         detail={
        #             "error_name": reply_from_llm.__class__.__name__,
        #             "error_msg": str(reply_from_llm)
        #         }
        #     )
        
        # updated_chat = await updateConversation(id, user_data, reply_from_llm)

        # Return the response in the required DTO format
        # return reply_from_llm
        return StreamingResponse(response_from_LLM(query=user_data.new_message), media_type="text/event_stream")

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

    
    return "OKg"





# @router.delete("/{id}")
# def delete_item(
#     session, current_user, id: uuid.UUID
# ) -> Message:
#     """
#     Delete an item.
#     """
#     item = session.get(Item, id)
#     if not item:
#         raise HTTPException(status_code=404, detail="Item not found")
#     if not current_user.is_superuser and (item.owner_id != current_user.id):
#         raise HTTPException(status_code=400, detail="Not enough permissions")
#     session.delete(item)
#     session.commit()
#     return Message(message="Item deleted successfully")
