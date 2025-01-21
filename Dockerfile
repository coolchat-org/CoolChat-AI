# Base image có hỗ trợ Python và các công cụ cần thiết
FROM ubuntu:20.04

# Cập nhật hệ thống và cài đặt LibreOffice
RUN apt-get update && apt-get install -y \
    libreoffice \
    python3 \
    python3-pip && \
    apt-get clean

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn vào container
COPY . .

# Cài đặt các thư viện Python
RUN pip3 install .

# Expose cổng ứng dụng (thay 8000 bằng cổng mà FastAPI đang dùng)
EXPOSE 8000

# Lệnh khởi động ứng dụng
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
