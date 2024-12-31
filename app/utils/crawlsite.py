from bs4 import BeautifulSoup
# import requests
# from requests import Response
import httpx
import asyncio
from typing import List
from langchain_core.documents import Document
# from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

async def fetch_content(url: str) -> str:
    """
    Fetch the content of a single website asynchronously.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
    
async def crawl_sites(websites: List[str]) -> List[Document]:
    """
    Crawl content from a list of websites and return LangChain Document objects.
    """
    documents: List[Document] = []
    meaningful_tags: set[str] = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "article", "section", "blockquote", "a", "div", "span"}

    async def process_site(url: str):
        # fetch content
        try:
            html_content = await fetch_content(url)

            # parse the content
            soup = BeautifulSoup(html_content, "html.parser")

            # remove unwanted tag
            for tag in soup(["footer", "nav", "script", "style", "aside"]):
                tag.decompose()

            # extract content from h1 to the rest of the body
            h1_encountered: bool = False
            content: str = ""
            for element in soup.body.descendants:
                if element.name == "h1":
                    h1_encountered = True
                if h1_encountered and element.name in meaningful_tags:
                    text = element.get_text(strip=True)
                    if element.name == "a":
                        text = text + " " + element['href']
                    if text:
                        content = content + " " + text
            # Create a LangChain Document object
            if content:
                documents.append(Document(page_content=content, metadata={"source": url}))

        except Exception as e:
            print(f"Error while processing {url}: {e}")

    # Use asyncio.gather with a progress bar
    await tqdm_asyncio.gather(*(process_site(url) for url in websites), desc="Crawling websites", unit="site")

    return documents



