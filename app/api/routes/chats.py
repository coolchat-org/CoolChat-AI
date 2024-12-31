import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("", response_model=str)
def read_items() -> Any:
    """
    Retrieve items.
    """
    
    return "OK"


@router.post("/{id}", response_model=str)
def createReplyMsg() -> Any:
    """
    Get item by ID.
    """
    
    return "OK"


@router.post("/create-index", response_model=str)
def createLocalIndex() -> Any:
    """
    Create new item.
    """
    
    return "ok"


@router.put("/new-chat", response_model=str)
def createNewChat() -> Any:
    """
    Update an item.
    """
    
    return "ok"


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
