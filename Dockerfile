# Sử dụng Ubuntu làm base image
FROM ubuntu:22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
# ENV PATH="$HOME/.local/bin:$PATH"
ENV PATH="/root/.local/bin:$PATH"

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tzdata \
    libreoffice \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt UV Package Manager
RUN wget -qO- https://astral.sh/uv/install.sh | sh

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn vào container
COPY . .

# Đồng bộ dependencies qua UV
RUN uv sync

# Expose cổng ứng dụng
EXPOSE 8000

# Lệnh khởi động ứng dụng
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
