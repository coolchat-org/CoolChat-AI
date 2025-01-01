import os
from typing import Any
from fastapi import APIRouter, HTTPException, status
from app.api.services.chatService import createIndexesFromFiles
from app.dtos.chatUserDto import CreateIndexDto

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("", response_model=str)
def read_items() -> Any:
    """
    Retrieve items.
    """
    
    return "OK"

@router.post("/create-index", response_model=CreateIndexDto)
async def createLocalIndex() -> Any:
    """
    Create new item.
    """
    try:
        # demo folder path
        relative_folder: str = os.path.join(os.pardir, os.pardir, "docs")  # Lùi 2 cấp vào thư mục "docs"
        exact_folder: str = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_folder))
        print("Exact folder: ", exact_folder)

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


@router.post("/{id}", response_model=str)
def createReplyMsg() -> Any:
    """
    Get item by ID.
    """
    
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
