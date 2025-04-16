# syntax=docker/dockerfile:1.4
FROM ubuntu:22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1  
ENV PYTHONDONTWRITEBYTECODE=1  

# Cài đặt các gói cần thiết và dependencies xây dựng
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tzdata \
    python3 \
    python3-pip \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt UV Package Manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file cấu hình cài đặt dependencies trước để tận dụng cache
COPY pyproject.toml uv.lock ./

# Đồng bộ dependencies qua UV sử dụng cache (BuildKit phải được bật)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --platform linux_x86_64

# Copy toàn bộ mã nguồn còn lại
COPY . .

# Expose cổng ứng dụng
EXPOSE 8000

# Lệnh khởi động ứng dụng với Gunicorn
CMD [".venv/bin/gunicorn", "app.server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]
