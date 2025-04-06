from fastapi import UploadFile


def get_file_type(file: UploadFile) -> str:
    if file.filename.endswith(".pdf") or file.content_type == "application/pdf":
        return "pdf"
    elif file.filename.endswith(".docx") or file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    elif file.filename.endswith(".txt") or "text/plain" in file.content_type:
        return "txt"
    else:
        return "unsupported"