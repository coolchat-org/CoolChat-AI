import hashlib
import logging
import os
import tempfile
import shutil
import subprocess
import time
from typing import Any, Dict, List, Tuple
from uuid import uuid4
from charset_normalizer import detect
from fastapi import HTTPException, UploadFile, logger, status
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from app.dtos.chatUserDto import CreateIndexDto
from app.utils.crawlsite import crawl_sites
from app.core.config import settings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from fastapi import UploadFile
import os
import tempfile
import shutil
import subprocess

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def convert_doc_to_docx(doc_file: UploadFile) -> str:
    """
    Convert a .doc file to .docx format.
    
    Args:
        doc_file (UploadFile): The uploaded .doc file.
        
    Returns:
        str: Path to the converted .docx file.
    """
    try:
        # Save the uploaded .doc file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as temp_doc:
            content = await doc_file.read()
            temp_doc.write(content)
            temp_doc_path = temp_doc.name

        # Create output path for the .docx file
        docx_file = tempfile.mktemp(suffix=".docx")

        # Use LibreOffice to convert the file
        command = [
            "soffice",
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            os.path.dirname(docx_file),
            temp_doc_path,
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {process.stderr.decode()}")

        # Ensure the output file exists
        converted_file_path = temp_doc_path.replace(".doc", ".docx")
        if not os.path.exists(converted_file_path):
            raise FileNotFoundError(f"Converted file not found: {converted_file_path}")
        
        shutil.move(converted_file_path, docx_file)  # Move to final path
        return docx_file

    except Exception as e:
        raise RuntimeError(f"Error converting file: {e}")



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

        if file.filename.endswith(".doc") and loader_cls == Docx2txtLoader:
            # Convert .doc to .docx
            print(f"Converting .doc to .docx for {file}...")
            docx_file = await convert_doc_to_docx(file)
            loader = loader_cls(docx_file)
            documents = loader.load()
            os.remove(docx_file)  # Clean up temporary file
        else:
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
        chunk_size=1000, chunk_overlap=50,
        length_function=len, is_separator_regex=False)
    if length > 1000:
        batch_size = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,          # Tăng chunk_size để giảm số lượng đoạn
            chunk_overlap=300,        # Tăng chunk_overlap nếu cần ngữ cảnh rộng hơn
            length_function=len,
            is_separator_regex=False
        )
    elif length > 500:
        batch_size = 20
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,          # Increase chunk_size => decrease num of paragraph
            chunk_overlap=100,        # Increase chunk_overlap for larger context
            length_function=len,
            is_separator_regex=False
        )

    return batch_size, text_splitter

def generate_index_name(organization_id: str, prefix: str = "org-index") -> str:
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
        # --- Check index exists
        if len(pdfFiles) == 0 and len(docxFiles) == 0 and len(txtFiles) == 0 and len(websites_data) == 0:
            raise ValueError("No valid files (PDF, Word, or TXT) or websites found.")
        else:
            print("Found pdf files: ", len(pdfFiles))
            print("Found doc files: ", len(docxFiles))
            print("Found txt files: ", len(txtFiles))
            print("Found sites: ", len(websites_data))
    
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        # if organization_id.endswith("_0"):
        #     organization_id = organization_id[:-2]
        index_name = generate_index_name(organization_id=organization_id)

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name in existing_indexes:
            # directly create
            if not is_virtual:
                # do not raise exception, delete it
                pc.delete_index(name=index_name, timeout=10)
            else:
                index_name = index_name + "-0"

        elif index_name + "-0" in existing_indexes:
            if not is_virtual:
                # do not raise exception, delete it
                pc.delete_index(name=index_name + "-0", timeout=10)
        
        
        normalDocs: List[Document] = await load_files_by_type(
            files=pdfFiles + docxFiles + txtFiles,
            priorities=pdfPrio + docxPrio + txtPrio
        )
        webDocs: List[Document] = await crawl_sites(websites_data=websites_data)

        # allDocs: List[Document] = normalDocs + webDocs
        # allDocs = [extract_metadata(doc) for doc in (normalDocs + webDocs)]
        allDocs: List[Document] = normalDocs + webDocs

        print(f'Total documents loaded: {len(allDocs)}')

        batch_size, text_splitter = set_batch_and_splitter(len(allDocs))
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
        
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)
        print("Processing and indexing documents...")
        vector_store = PineconeVectorStore(index=index, embedding=embedder)
        for i in tqdm(range(0, len(allDocs), batch_size), desc="Processing batches", unit="batch"):
            batch = allDocs[i:i + batch_size]
            batch_splits = text_splitter.split_documents(batch)
            await vector_store.aadd_documents(batch_splits) 
        
        index_url = pc.describe_index(index_name)["host"]

        return CreateIndexDto(
            message="Index creation completed successfully.",
            index_name=index_name,
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


def saveVirtualIndex(old_index_name: str, organization_id: str = "12345xoz"):
    try:
        # --- Check index exists
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name_1 = generate_index_name(organization_id=organization_id)
        index_name_2 = index_name_1 + "-0"
        if old_index_name != index_name_1 and old_index_name != index_name_2:
            raise Exception(f"Bad organization id entered!")
        
        if old_index_name.endswith("-0"):
            index_to_save = index_name_1
        else:
            index_to_save = index_name_2

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if old_index_name not in existing_indexes:
            raise Exception(f"Index '{old_index_name}' did not exists.")
        elif index_to_save not in existing_indexes:
            raise Exception(f"Index '{index_to_save}' did not exists.")
        
        pc.delete_index(name=old_index_name, timeout=10)        
        index_url = pc.describe_index(index_to_save)["host"]

        return CreateIndexDto(
            message="Virtual Index saving completed successfully.",
            index_name=index_to_save,
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

def disposeVirtualIndex(old_index_name: str, organization_id: str = "12345xoz"):
    try:
        # --- Check index exists
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name_1 = generate_index_name(organization_id=organization_id)
        index_name_2 = index_name_1 + "-0"
        if old_index_name != index_name_1 and old_index_name != index_name_2:
            raise Exception(f"Bad organization id entered!")
        
        if old_index_name.endswith("-0"):
            index_to_delete = index_name_1
        else:
            index_to_delete = index_name_2

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if old_index_name not in existing_indexes:
            raise Exception(f"Index '{old_index_name}' did not exists.")
        elif index_to_delete not in existing_indexes:
            raise Exception(f"Index '{index_to_delete}' did not exists.")
        
        pc.delete_index(name=index_to_delete, timeout=10)        

        return CreateIndexDto(
            message="Virtual Index deleting completed successfully.",
            index_name="",
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

