from typing import List, Dict
from langchain_core.documents import Document
from tqdm.asyncio import tqdm_asyncio
from app.core.config import settings
import asyncio
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from tqdm.asyncio import tqdm_asyncio
from langchain_core.documents import Document

async def load_site(url: str, priority: int, api_key: str) -> List[Document]:
    loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")

    try:
        documents = await asyncio.to_thread(loader.load)
        for doc in documents:
            doc.metadata["priority"] = priority
            doc.metadata["source"] = url
        return documents
    except Exception as e:
        print(f"Error while scraping {url}: {e}")
        return []

async def crawl_sites(websites_data: List[Dict]) -> List[Document]:
    tasks = [load_site(web["url"], web["priority"], settings.FIRECRAWL_API_KEY) for web in websites_data]
    results = await tqdm_asyncio.gather(*tasks, desc="Collecting web info...", unit="site")
    all_documents = [doc for res in results for doc in res if res is not None]
    return all_documents


# if __name__ == "__main__":
#     # For testing, try a few different websites including some JS-heavy ones
#     test_sites = [
#         {"url": "https://itviec.com/blog/faq-chuc-nang-loi-moi-cong-viec-danh-cho-ung-vien/", "priority": 1},
#         {"url": "https://itviec.com/blog/faq-ve-chuc-nang-company-review/", "priority": 1},  # React docs
#         # {"url": "https://vuejs.org/guide/introduction.html", "priority": 1},  # Vue docs
#         # {"url": "https://angular.io/guide/what-is-angular", "priority": 1},  # Angular docs
#     ]
    
#     documents = asyncio.run(crawl_sites(test_sites))
    
#     print("\n=== Crawl Results ===")
#     for i, doc in enumerate(documents):
#         print(f"\nDocument {i+1}:")
#         print(f"Source: {doc.metadata['source']}")
#         print(f"Title: {doc.metadata.get('title', 'No title')}")
#         print(f"Content Length: {len(doc.page_content)} characters")
#         print(f"Content Preview: {doc.page_content}...")


















# from typing import List, Dict, Any, Set
# import asyncio
# from langchain_core.documents import Document
# from bs4 import BeautifulSoup
# from playwright.async_api import async_playwright
# from tqdm.asyncio import tqdm_asyncio
# import re
# import logging

# # We cant switch to FireCrawl when large-scale

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# async def fetch_content_with_js(url: str, timeout: int = 60000, retry_count: int = 2) -> str:
#     """
#     Fetch content from a URL with JavaScript rendering support.
    
#     Args:
#         url: The URL to fetch
#         timeout: Timeout in milliseconds
#         retry_count: Number of retries on failure
        
#     Returns:
#         The HTML content as a string
#     """
#     for attempt in range(retry_count + 1):
#         try:
#             logger.info(f"Fetching {url} (attempt {attempt+1}/{retry_count+1})")
            
#             async with async_playwright() as p:
#                 # Launch with more options for stability
#                 browser = await p.chromium.launch(
#                     headless=True,
#                     args=[
#                         '--disable-gpu',
#                         '--disable-dev-shm-usage',
#                         '--disable-setuid-sandbox',
#                         '--no-sandbox',
#                         '--disable-web-security',  # Helps with CORS issues
#                         '--disable-features=IsolateOrigins,site-per-process'
#                     ]
#                 )
                
#                 # Create context with more realistic browser behavior
#                 context = await browser.new_context(
#                     viewport={"width": 1920, "height": 1080},
#                     user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#                     ignore_https_errors=True,  # Ignore SSL errors
#                     java_script_enabled=True,  # Ensure JavaScript is enabled
#                     accept_downloads=False
#                 )
                
#                 # Add timeout handling
#                 page = await context.new_page()
                
#                 try:
#                     # Use domcontentloaded instead of networkidle for faster initial load
#                     response = await page.goto(
#                         url, 
#                         timeout=timeout,
#                         wait_until="domcontentloaded"  # Less strict than networkidle
#                     )
                    
#                     if not response:
#                         logger.warning(f"No response from {url}")
#                         if attempt < retry_count:
#                             continue
#                         return ""
                        
#                     if response.status >= 400:
#                         logger.error(f"Failed to load {url}: HTTP {response.status}")
#                         if attempt < retry_count:
#                             continue
#                         return ""
                    
#                     # Wait for important content to load
#                     try:
#                         # Wait for common content indicators with shorter timeout
#                         content_selectors = ["main", "#root", "#app", ".content", "article", "p", "h1"]
#                         for selector in content_selectors:
#                             try:
#                                 await page.wait_for_selector(selector, timeout=5000)
#                                 break  # Stop once we find one selector
#                             except:
#                                 continue
#                     except:
#                         # Continue even if we couldn't find any of these selectors
#                         pass
                    
#                     # Try to wait for network idle with a shorter timeout
#                     try:
#                         await page.wait_for_load_state("networkidle", timeout=10000)
#                     except:
#                         logger.info(f"Network didn't reach idle state for {url}, proceeding anyway")
                    
#                     # Scroll to load lazy content
#                     await scroll_page(page)
                    
#                     # Get the fully rendered HTML
#                     content = await page.content()
                    
#                     await browser.close()
                    
#                     if content:
#                         return content
#                     else:
#                         logger.warning(f"Empty content from {url}")
#                         if attempt < retry_count:
#                             continue
#                         return ""
                        
#                 except Exception as e:
#                     await browser.close()
#                     logger.error(f"Navigation error for {url}: {str(e)}")
#                     if attempt < retry_count:
#                         logger.info(f"Retrying {url}...")
#                         continue
#                     return ""
                    
#         except Exception as e:
#             logger.error(f"Browser error fetching {url}: {str(e)}")
#             if attempt < retry_count:
#                 logger.info(f"Retrying {url}...")
#                 continue
#             return ""
    
#     return ""  # Return empty string if all attempts fail

# async def scroll_page(page):
#     """Scroll the page progressively to trigger lazy loading."""
#     try:
#         # Get page height
#         height = await page.evaluate("document.body.scrollHeight")
        
#         # Scroll in chunks
#         viewport_height = await page.evaluate("window.innerHeight")
#         view_portions = min(10, max(1, int(height / viewport_height)))
        
#         for i in range(view_portions):
#             scroll_position = int(height * (i / view_portions))
#             await page.evaluate(f"window.scrollTo(0, {scroll_position})")
#             await asyncio.sleep(0.2)  # Short pause between scrolls
        
#         # Final scroll to bottom
#         await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
#         await asyncio.sleep(0.5)
        
#     except Exception as e:
#         logger.warning(f"Error during page scrolling: {str(e)}")

# async def crawl_sites(websites_data: List[Dict[str, Any]]) -> List[Document]:
#     """
#     Crawl content from a list of websites and return LangChain Document objects.
#     Supports JavaScript-rendered websites like React, Vue, and Angular.
#     """
#     documents: List[Document] = []
#     if len(websites_data) == 0:
#         return documents
    
#     # Expanded set of meaningful tags to better capture JS-based sites
#     meaningful_tags: Set[str] = {
#         "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "article", "section", 
#         "blockquote", "a", "div", "span", "main", "strong", "em", "code", 
#         "pre", "table", "td", "th", "tr", "dl", "dt", "dd", "ul", "ol"
#     }
    
#     # Common selectors for content in popular frameworks
#     content_selectors = [
#         "main", "#root", "#app", ".content", "article", 
#         "[role='main']", ".main-content", ".page-content",
#         ".container", ".wrapper", "#content", ".article-content"
#     ]

#     async def process_site(url: str, priority: int):
#         try:
#             # Try to handle URLs with potential typos or missing protocol
#             if not url.startswith(('http://', 'https://')):
#                 url = f"https://{url}"
                
#             # Fetch content with JavaScript rendering - longer timeout and retry
#             html_content = await fetch_content_with_js(url, timeout=60000, retry_count=2)
#             if not html_content:
#                 logger.warning(f"No content retrieved from {url}")
#                 return

#             # Parse the content
#             soup = BeautifulSoup(html_content, "html.parser")

#             # Remove unwanted elements
#             for tag in soup(["nav", "script", "style", "aside", "noscript", 
#                            ".cookie-banner", ".advertisement", ".ads", ".nav", ".menu",
#                            ".sidebar", ".comments", "[role='banner']", "[role='navigation']",
#                            "iframe", "svg", "button", "form", "input"]):
#                 tag.decompose()

#             # Try to identify main content area
#             main_content = None
#             for selector in content_selectors:
#                 try:
#                     content_areas = soup.select(selector)
#                     for content_area in content_areas:
#                         if content_area and len(content_area.get_text(strip=True)) > 200:
#                             main_content = content_area
#                             break
#                     if main_content:
#                         break
#                 except:
#                     continue
            
#             # If no main content area found, use the body
#             content_area = main_content if main_content else soup.body
            
#             # Extract content
#             h1_encountered: bool = False
#             content: List[str] = []
            
#             if content_area:
#                 # Get title if available
#                 title = soup.title.string if soup.title else ""
#                 if title:
#                     content.append(f"Title: {title}")
                
#                 for element in content_area.descendants:
#                     # Start collecting after the first h1 (usually the title)
#                     if element.name == "h1":
#                         h1_encountered = True
#                         content.append(f"Heading: {element.get_text(strip=True)}")
                    
#                     if (h1_encountered or not soup.find('h1')) and element.name in meaningful_tags:
#                         text = element.get_text(strip=True)
                        
#                         # Add context for headings
#                         if element.name and element.name.startswith('h') and len(element.name) == 2:
#                             text = f"{element.name.upper()}: {text}"
                        
#                         # Handle anchor tags (add href)
#                         if element.name == "a" and element.get("href"):
#                             href = element["href"]
#                             # Make relative URLs absolute
#                             if not href.startswith(('http://', 'https://')):
#                                 if href.startswith('/'):
#                                     base_url = re.match(r'(https?://[^/]+)', url)
#                                     if base_url:
#                                         href = f"{base_url.group(1)}{href}"
#                                 else:
#                                     href = f"{url.rstrip('/')}/{href.lstrip('/')}"
#                             text = f"{text} ({href})"

#                         if text and len(text) > 1:  # Avoid single characters
#                             content.append(text)
                
#                 # Create a LangChain Document object
#                 if content:
#                     full_content = " ".join(content)
#                     # Clean up excessive whitespace
#                     full_content = re.sub(r'\s+', ' ', full_content).strip()
                    
#                     if full_content and len(full_content) > 50:  # Ensure we have meaningful content
#                         documents.append(
#                             Document(
#                                 page_content=full_content, 
#                                 metadata={
#                                     "source": url, 
#                                     "priority": priority,
#                                     "title": soup.title.string if soup.title else url
#                                 }
#                             )
#                         )
#                         logger.info(f"Successfully extracted content from {url}: {len(full_content)} characters")
#                     else:
#                         logger.warning(f"Content too short from {url}: {len(full_content) if full_content else 0} characters")
#                 else:
#                     logger.warning(f"No content extracted from {url}")
#             else:
#                 logger.warning(f"No content area found in {url}")

#         except Exception as e:
#             logger.error(f"Error while processing {url}: {str(e)}")

#     # Set a concurrency limit to avoid overloading the browser
#     semaphore = asyncio.Semaphore(3)  # Reduce to 3 concurrent sites for stability
    
#     async def bounded_process_site(web_data):
#         async with semaphore:
#             await process_site(web_data["url"], web_data["priority"])
    
#     # Use asyncio.gather with a progress bar and concurrency control
#     await tqdm_asyncio.gather(
#         *(bounded_process_site(web_data) for web_data in websites_data), 
#         desc="Crawling websites", 
#         unit="site"
#     )

#     logger.info(f"Successfully crawled {len(documents)} documents from {len(websites_data)} websites")
#     return documents


# if __name__ == "__main__":
#     # For testing, try a few different websites including some JS-heavy ones
#     test_sites = [
#         {"url": "https://hcmut.edu.vn/", "priority": 1},
#         # {"url": "https://reactjs.org/", "priority": 1},  # React docs
#         # {"url": "https://vuejs.org/guide/introduction.html", "priority": 1},  # Vue docs
#         # {"url": "https://angular.io/guide/what-is-angular", "priority": 1},  # Angular docs
#     ]
    
#     documents = asyncio.run(crawl_sites(test_sites))
    
#     print("\n=== Crawl Results ===")
#     for i, doc in enumerate(documents):
#         print(f"\nDocument {i+1}:")
#         print(f"Source: {doc.metadata['source']}")
#         print(f"Title: {doc.metadata.get('title', 'No title')}")
#         print(f"Content Length: {len(doc.page_content)} characters")
#         print(f"Content Preview: {doc.page_content}...")
