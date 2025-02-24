from typing import List
from langchain_core.documents import Document

def extract_metadata(doc: Document):
    print(doc.metadata)
    priority = doc.metadata.get("priority", "1")
    return Document(
        page_content=doc.page_content,
        metadata={
            "priority": priority,
            "source": doc.metadata.get("source", "unknown")
        }
    )

# from docx import Document
# from PyPDF2 import PdfReader

# def extract_metadata_docx(file):
#     doc = Document(file)
#     props = doc.core_properties
#     return {
#         "title": props.title,
#         "subject": props.subject,
#         "keywords": props.keywords,
#         "comments": props.comments,
#     }
# def print_metadata_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     metadata = reader.metadata  # Lấy metadata
    
#     print(f"Metadata của PDF: {pdf_path}")
#     for key, value in metadata.items():
#         print(f"{key}: {value}")


# print(extract_metadata_docx("D:/FastAPILLM/app/docs/modified_New Doc.docx"))
# # print_metadata_pdf("D:/FastAPILLM/app/docs/modified_2106.09685v2.pdf")
# priorities: List[int] = [
#     4, 2, 3, 1, 0
# ] 
# scaler = MinMaxScaler()
# scaler.fit(priorities)
# priorities = scaler.transform(priorities)
# print(priorities)
# results = [("a", 3), ("b", 2), ("c", 6)]
# priorities: List[int] = [
#             1, 2, 3
#         ] 
# max_prio, min_prio = max(priorities), min(priorities)
# if max_prio == min_prio:
#     scaled_priorities = [1 for _ in range(len(priorities))]
# else:
#     scaled_priorities = [(x - min_prio) / (max_prio - min_prio) for x in priorities]

# sorted_results = sorted(
#             enumerate(results), 
#             key=lambda elem: scaled_priorities[elem[0]] * 0.7 + elem[1][1] * 0.3, 
#             reverse=True
#         )

# sorted_results = [elem[1][0] for elem in sorted_results]
# print(sorted_results)