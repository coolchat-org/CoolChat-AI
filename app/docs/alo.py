import os
from PyPDF2 import PdfReader, PdfWriter
from docx import Document

# Hàm để chèn metadata vào PDF
def embed_metadata_pdf(pdf_path, output_pdf_path, metadata):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Sao chép tất cả các trang vào writer
    for page in reader.pages:
        writer.add_page(page)
    
    # Chèn metadata vào
    writer.add_metadata(metadata)
    
    # Lưu file PDF mới
    with open(output_pdf_path, "wb") as output_pdf:
        writer.write(output_pdf)

# Hàm để chèn metadata vào DOCX
def embed_metadata_docx(docx_path, output_docx_path, metadata):
    doc = Document(docx_path)

    # Thêm metadata vào phần thông tin tài liệu
    core_props = doc.core_properties
    for key, value in metadata.items():
        if hasattr(core_props, key):
            setattr(core_props, key, value)

    # Lưu file DOCX mới
    doc.save(output_docx_path)

# Hàm để tải lên tất cả các file trong thư mục và thêm metadata
def process_files(folder_path, metadata):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Nếu là file PDF
        if filename.lower().endswith('.pdf'):
            metadata = metadata_pdf
            output_pdf_path = os.path.join(folder_path, f"modified_{filename}")
            embed_metadata_pdf(file_path, output_pdf_path, metadata)
            print(f"Đã thêm metadata vào file PDF: {filename}")
        
        # Nếu là file DOCX
        elif filename.lower().endswith('.docx'):
            metadata = metadata_docx
            output_docx_path = os.path.join(folder_path, f"modified_{filename}")
            embed_metadata_docx(file_path, output_docx_path, metadata)
            print(f"Đã thêm metadata vào file DOCX: {filename}")

# Metadata cần chèn vào các file
metadata_docx = {
    'subject': '3'
}
metadata_pdf = {
    '/Priority': '5'
}

# Thư mục chứa các file cần xử lý
folder_path = 'path_to_your_folder'  # Thay thế bằng đường dẫn đến thư mục của bạn
# process_files(folder_path, metadata)
if __name__ == "__main__":
    process_files("D:/FastAPILLM/app/docs", None)
