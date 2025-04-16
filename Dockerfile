FROM ubuntu:22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1  
ENV PYTHONDONTWRITEBYTECODE=1  
ENV GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
ENV GRPC_PYTHON_BUILD_WITH_CYTHON=1

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tzdata \
    python3 \
    python3-pip \
    # Build dependencies
    build-essential \
    python3-dev \
    # Dependencies cho grpcio
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

# Sao chép chỉ các file cần thiết để cài đặt dependencies
COPY pyproject.toml .
COPY uv.lock .  

# # Đồng bộ dependencies qua UV với build cache
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --platform linux_x86_64

# Sao chép toàn bộ mã nguồn
COPY . .

# Tạo user không phải root để tăng bảo mật
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Expose cổng ứng dụng
EXPOSE 8000

# Lệnh khởi động ứng dụng với Gunicorn
CMD [".venv/bin/gunicorn", "app.server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]