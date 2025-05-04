import asyncio
import hashlib
import logging
import os
import tempfile
from typing import Any, Dict, List, Tuple
from uuid import uuid4
from charset_normalizer import detect
from fastapi import HTTPException, UploadFile, logger, status
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from app.dtos.indexDto import CreateNamespaceDto
from app.utils.crawlsite import crawl_sites
from app.core.config import settings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from fastapi import UploadFile


# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_file_encoding(file: UploadFile) -> str:
    """
    Detect the encoding of a text file.
    """
    with open(file, "rb") as f:
        result = detect(f.read())
        return result["encoding"]
    
# from langchain_core.documents import Document
async def process_file(file: UploadFile, priority: int) -> List[Document]:
    """
    Process a single file asynchronously.
    """
    try:
        loader_cls = Docx2txtLoader 
        if file.content_type == "application/pdf":
            loader_cls = PyPDFLoader
        elif file.content_type in ["text/plain; charset=utf-8", "text/plain"]:
            loader_cls = TextLoader

        # Save the file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename.split('.')[-1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Load the file using the specified loader
        if loader_cls == TextLoader:
            encoding = detect_file_encoding(temp_file_path)  # Ensure this function takes a file path
            loader = loader_cls(temp_file_path, encoding=encoding)
        else:
            loader = loader_cls(temp_file_path)
        documents = loader.load()
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["priority"] = priority

        # Clean up temporary file
        os.remove(temp_file_path)

        return documents
    
    except Exception as error:
        print(f"Error processing file {file}: {error}")
        return []
    

async def load_files_by_type(files: List[UploadFile], priorities: List[int]) -> List[Document]:
    """
    Load files of a specific type and return a list of LangChain Document objects.

    Args:
        files (List[UploadFile]): List of file paths.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    
    tasks = [process_file(file, priority) for file, priority in zip(files, priorities)]
    results = await tqdm_asyncio.gather(*tasks, desc="Loading files", unit="file")
    # flatten the list of document lists
    all_documents = [doc for documents in results for doc in documents]

    # print(f"Loaded {len(all_documents)} documents.")
    return all_documents

def set_batch_and_splitter(length: int) -> Tuple[int, RecursiveCharacterTextSplitter]:
    """
    Return the proper batch and TextSplitter with right chunk size & overlap
    """
    batch_size = 10
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        length_function=len, is_separator_regex=False)
    if length > 1000:
        batch_size = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,          # Tăng chunk_size để giảm số lượng đoạn
            chunk_overlap=300,        # Tăng chunk_overlap nếu cần ngữ cảnh rộng hơn
            length_function=len,
            is_separator_regex=False
        )
    elif length > 500:
        batch_size = 20
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,          # Increase chunk_size => decrease num of paragraph
            chunk_overlap=200,        # Increase chunk_overlap for larger context
            length_function=len,
            is_separator_regex=False
        )

    return batch_size, text_splitter

def generate_org_name(organization_id: str, prefix: str) -> str:
    """
    Generate a unique Pinecone index name using SHA-256 hash of the organization_id.

    Args:
        organization_id (str): The organization ID to hash.
        prefix (str): Optional prefix for the index name (default is 'org_index').

    Returns:
        str: A unique index name.
    """
    # Hash the organization_id using SHA-256
    hash_object = hashlib.sha256(organization_id.encode())
    hashed_id = hash_object.hexdigest()  # Get the hexadecimal representation of the hash

    # Combine prefix and hashed ID to create the index name
    index_name = f"{prefix}-{hashed_id[:16]}"  # Use the first 16 characters of the hash for brevity
    return index_name


# main function use camelCase
async def createIndexesFromFilesAndUrls(websites_data: List[Dict[str, Any]], 
    pdfFiles: List[UploadFile] = None, docxFiles: List[UploadFile] = None, txtFiles: List[UploadFile] = None, pdfPrio: List[int] = [], docxPrio: List[int] = [], txtPrio: List[int] = [], 
    organization_id: str = "12345xoz", is_virtual: bool = False):
    try:
        # --- Check valid data
        if not any([pdfFiles, docxFiles, txtFiles, websites_data]):
            raise ValueError("No valid files (PDF, Word, or TXT) or websites provided.")
        
        # --- Check shared index exists
        shared_index_name = settings.INDEX_NAME
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if shared_index_name not in existing_indexes:
            pc.create_index(
                name=shared_index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Creating shared index: {shared_index_name}")
            while not pc.describe_index(shared_index_name).status["ready"]:
                await asyncio.sleep(1)
        else:
            print(f"Shared index '{shared_index_name}' existed.")

        # --- Handle org namespace
        org_namespace = generate_org_name(organization_id, prefix=shared_index_name)    

        index = pc.Index(shared_index_name)
        index_stats = index.describe_index_stats()
        existing_namespaces = list(index_stats.get("namespaces", {}).keys())
        # print("List namespace: ", existing_namespaces)
        if org_namespace in existing_namespaces:
            if not is_virtual:
                index.delete(delete_all=True, namespace=org_namespace)
            else:
                org_namespace = org_namespace + "-0"
        elif org_namespace + "-0" in existing_indexes:
            if not is_virtual:
                # do not raise exception, delete it
                index.delete(delete_all=True, namespace=org_namespace + "-0")
        
        # --- Main code: load and index file
        normalDocs: List[Document] = await load_files_by_type(
            files=pdfFiles + docxFiles + txtFiles,
            priorities=pdfPrio + docxPrio + txtPrio
        )
        webDocs: List[Document] = await crawl_sites(websites_data=websites_data)
        allDocs: List[Document] = normalDocs + webDocs

        print(f'Total documents loaded: {len(allDocs)}')

        batch_size, text_splitter = set_batch_and_splitter(len(allDocs))
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
        # embedder = GoogleGenerativeAIEmbeddings(
        #     model_name="gemini-embedding-exp-03-07", task_type="RETRIEVAL_DOCUMENT", 
        # )

        print("Processing and indexing documents...")
        vector_store = PineconeVectorStore(index=index, embedding=embedder)
        for i in tqdm(range(0, len(allDocs), batch_size), desc="Processing batches", unit="batch"):
            batch = allDocs[i:i + batch_size]
            batch_splits = text_splitter.split_documents(batch)
            await vector_store.aadd_documents(batch_splits, namespace=org_namespace) 
        
        index_url = pc.describe_index(shared_index_name)["host"]

        return CreateNamespaceDto(
            message="Index creation completed successfully.",
            namespace=org_namespace,
            index_url=index_url,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


def saveVirtualIndex(old_namespace: str, organization_id: str = "12345xoz"):
    try:
        # --- Check index exists
        shared_index_name = settings.INDEX_NAME
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(shared_index_name)
        
        # --- Check namespace exists
        namespace_1 = generate_org_name(organization_id=organization_id, prefix=shared_index_name)
        namespace_2 = namespace_1 + "-0"
        if old_namespace != namespace_1 and old_namespace != namespace_2:
            raise Exception(f"Bad organization id entered!")

        
        if old_namespace.endswith("-0"):
            namespace_to_save = namespace_1
        else:
            namespace_to_save = namespace_2

        index_stats = index.describe_index_stats()
        existing_namespaces = list(index_stats.get("namespaces", {}).keys())
        print("List namespace: ", existing_namespaces)
        if old_namespace not in existing_namespaces:
            raise Exception(f"Namespace to delete: '{old_namespace}' did not exists.")
        elif namespace_to_save not in existing_namespaces:
            raise Exception(f"Namespace to save: '{namespace_to_save}' did not exists.")
        
        index.delete(delete_all=True, namespace=old_namespace)     
        index_url = pc.describe_index(shared_index_name)["host"]

        return CreateNamespaceDto(
            message="Virtual namespace saving completed successfully.",
            namespace=namespace_to_save,
            index_url=index_url,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

def disposeVirtualIndex(old_namespace: str, organization_id: str = "12345xoz"):
    try:
        # --- Check index exists
        shared_index_name = settings.INDEX_NAME
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(shared_index_name)
        
        # --- Check namespace exists
        namespace_1 = generate_org_name(organization_id=organization_id, prefix=shared_index_name)
        namespace_2 = namespace_1 + "-0"
        if old_namespace != namespace_1 and old_namespace != namespace_2:
            raise Exception(f"Bad organization id entered!")
        
        if old_namespace.endswith("-0"):
            namespace_to_delete = namespace_1
        else:
            namespace_to_delete = namespace_2

        index_stats = index.describe_index_stats()
        existing_namespaces = list(index_stats.get("namespaces", {}).keys())
        print("List namespace: ", existing_namespaces)
        if old_namespace not in existing_namespaces:
            raise Exception(f"Old namespace: '{old_namespace}' did not exists.")
        elif namespace_to_delete not in existing_namespaces:
            raise Exception(f"Namespace to delete '{namespace_to_delete}' did not exists.")
        
        index.delete(delete_all=True, namespace=namespace_to_delete)         

        return CreateNamespaceDto(
            message="Virtual namespace deleting completed successfully.",
            namespace="",
            index_url="",
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

