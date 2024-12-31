import os
from re import split
import shutil
import tempfile
from typing import Any, List, Type
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from app.utils.crawlsite import crawl_sites
from app.core.config import settings


# Helper function use snake_case
def filter_files_by_extensions(folder_path: str, extensions: list[str]) -> list[str]:
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if any(file.endswith(ext) for ext in extensions)
    ]

async def convert_doc_to_docx(doc_file: str) -> str:
    """
    Convert a .doc file to .docx format (stub function - implement conversion logic).
    """
    # Placeholder implementation: requires `libreoffice` or `unoconv` for actual conversion
    try:
        docx_file = tempfile.mktemp(suffix=".docx")
        shutil.copy(doc_file, docx_file)  # Simulate conversion by copying
        return docx_file
    except Exception as e:
        raise RuntimeError(f"Failed to convert {doc_file} to .docx: {e}")

def get_loader(file: str) -> Type[BaseLoader]:
    """
    Determine the appropriate loader class for a given file type.
    """
    loader_mapping = {
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }

    # Get file extension
    _, ext = os.path.splitext(file)
    ext = ext.lower()

    if ext in loader_mapping:
        return loader_mapping[ext]

    raise ValueError(f"No loader available for file type: {ext}")


async def process_file(file: str) -> List[Document]:
    """
    Process a single file asynchronously.
    """
    try:
        loader_cls = get_loader(file)

        if file.endswith(".doc") and loader_cls == "Docx2txtLoader":
            # Convert .doc to .docx
            print(f"Converting .doc to .docx for {file}...")
            docx_file = await convert_doc_to_docx(file)
            loader = loader_cls(docx_file)
            documents = loader.load()
            os.remove(docx_file)  # Clean up temporary file
        else:
            # Load the file using the specified loader
            loader = loader_cls(file)
            documents = loader.load()

        return documents
    
    except Exception as error:
        print(f"Error processing file {file}: {error}")
        return []
    

async def load_files_by_type(files: List[str]) -> List[Document]:
    """
    Load files of a specific type and return a list of LangChain Document objects.

    Args:
        files (List[str]): List of file paths.
        Loader (Type[BaseLoader]): The document loader class to use.
        file_type (str): Type of file being loaded (e.g., PDF, Word).

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    tasks = [process_file(file) for file in files]
    results = await tqdm_asyncio.gather(*tasks, desc="Loading files", unit="file")
    # flatten the list of document lists
    all_documents = [doc for documents in results for doc in documents]

    print(f"Loaded {len(all_documents)} documents.")
    return all_documents


# main function use camelCase
async def createIndexesFromFiles(folder: str, websites: List[str]):
    try:
        pdfFiles: List[Any] = filter_files_by_extensions(folder_path=folder, extensions=[".pdf"])
        docFiles: List[Any] = filter_files_by_extensions(folder_path=folder, extensions=[".docx"])
        txtFiles: List[Any] = filter_files_by_extensions(folder_path=folder, extensions=[".docx"])

        if len(pdfFiles) == 0 and len(docFiles.length) == 0 and len(txtFiles.length) == 0:
            raise ValueError("No valid files (PDF, Word, or TXT) found in the folder.")
        else:
            print("Found pdf files: ", len(pdfFiles))
            print("Found doc files: ", len(docFiles))
            print("Found txt files: ", len(txtFiles))
        
        normalDocs: List[Document] = await load_files_by_type(pdfFiles + docFiles + txtFiles)
        # docDocs: List[Document] = await load_files_by_type(docFiles)
        # txtDocs: List[Document] = await load_files_by_type(docFiles)
        # Crawl websites
        webDocs: List[Document] = await crawl_sites(websites=websites)
        allDocs: List[Document] = normalDocs + webDocs

        print(f'Total documents loaded: {len(allDocs)}')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        
        batch_size: int = 10
        batched_splits = []
        # batch processing to make it faster
        for i in tqdm(range(0, len(allDocs), batch_size), desc="Processing batches", unit="batch"):
            batch = allDocs[i:i + batch_size]
            batch_splits = [text_splitter.split_documents([doc]) for doc in batch]

            # Gộp kết quả vào danh sách tổng
            batched_splits.extend(doc for split in batch_splits for doc in split)
        
        print(f"Total chunks created: {len(batched_splits)}")

        embedder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY
        )
        vector_store = FAISS.from_documents(batched_splits, embedder)

        # Lưu trữ FAISS vào thư mục
        relative_folder: str = os.path.join(os.pardir, os.pardir, "indexes")
        save_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_folder))
        print(f"Saving index to: {save_dir}")
        vector_store.save_local(save_dir)

        print("Index saved successfully.")

    except Exception as e:
        print("Error create indexes: ", e)
