from pydantic import BaseModel


class CreateIndexDto(BaseModel):
    message: str
    index_name: str
    index_url: str

class CreateNamespaceDto(BaseModel):
    message: str
    namespace: str
    index_url: str

class SaveIndexDto(BaseModel):
    old_index_name: str

class SaveNamespaceDto(BaseModel):
    current_namespace: str