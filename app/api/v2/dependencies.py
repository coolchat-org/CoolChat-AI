from typing import Any, Dict, List
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel


class Metadata(BaseModel):
    files: Dict[str, int]
    urls: List[Dict[str, Any]] = []

class IndexPreparation(BaseModel):
    pdf_files: List[UploadFile]
    docx_files: List[UploadFile]
    txt_files: List[UploadFile]
    pdf_prio: List[int]
    docx_prio: List[int]
    txt_prio: List[int]
    urls: List[Dict[str, Any]]

async def prepare_index(files: List[UploadFile]) -> IndexPreparation:
    # Check metadata.json
    metafile = next((f for f in files if f.filename == "metadata.json"), None)
    if not metafile:
        raise HTTPException(status_code=400, detail="metadata.json is required")
    
    # File is available
    raw = await metafile.read()
    meta = Metadata.model_validate_json(raw)

    # Remove metadata file
    files = [f for f in files if f.filename != "metadata.json"]
    pdf_files, docx_files, txt_files = [], [], []
    pdf_prio, docx_prio, txt_prio = [], [], []

    for f in files:
        t = get_file_type(f)
        prio = meta.files.get(f.filename, 0)
        if t == "pdf":
            pdf_files.append(f); pdf_prio.append(prio)
        elif t == "docx":
            docx_files.append(f); docx_prio.append(prio)
        elif t == "txt":
            txt_files.append(f); txt_prio.append(prio)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {f.filename}")
    
    return IndexPreparation(
        pdf_files=pdf_files,
        docx_files=docx_files,
        txt_files=txt_files,
        pdf_prio=pdf_prio,
        docx_prio=docx_prio,
        txt_prio=txt_prio,
        urls=meta.urls
    )

def get_file_type(file: UploadFile) -> str:
    if file.filename.endswith(".pdf") or file.content_type == "application/pdf":
        return "pdf"
    elif file.filename.endswith(".docx") or file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    elif file.filename.endswith(".txt") or "text/plain" in file.content_type:
        return "txt"
    else:
        return "unsupported"
    
