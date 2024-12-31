import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.services.chatService import createIndexesFromFiles

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("", response_model=str)
def read_items() -> Any:
    """
    Retrieve items.
    """
    
    return "OK"

@router.post("/create-index", response_model=str)
async def createLocalIndex() -> Any:
    """
    Create new item.
    """
    print("CC")
    relative_folder: str = os.path.join(os.pardir, os.pardir, "docs")  # Back 3 lần và vào thư mục "docs"
    exact_folder: str = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_folder))
    print("Exact folder: ", exact_folder)

    print("Exact folder: ", exact_folder)
    web_url: str = "https://vnexpress.net/than-thanh-hoa-ielts-4627600.html?utm_source=facebook&utm_medium=fanpage_VnE&utm_campaign=phuonguyen&fbclid=IwAR342qeRaOfJRwnJUl145GW3ojO2-S2XGcHa1XvxSqMTc6mKplJE2siI_qE"

    await createIndexesFromFiles(exact_folder, [web_url])
    return "ok"


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
