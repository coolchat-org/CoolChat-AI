FROM ubuntu:22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1  
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PORT=8000

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tzdata \
    python3 \
    python3-pip \
    # Thêm build dependencies
    build-essential \
    python3-dev \
    # Các dependencies cho Playwright
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxcb1 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libxcursor1 \
    libxtst6 \
    libxshmfence-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt UV Package Manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép chỉ các file cần thiết để cài đặt dependencies
COPY pyproject.toml .
COPY uv.lock .  

# Đồng bộ dependencies qua UV
RUN uv sync --frozen 

# Sao chép toàn bộ mã nguồn
COPY . .

# Tạo user không phải root để tăng bảo mật
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Expose cổng ứng dụng
EXPOSE ${PORT}

# Start application
CMD [".venv/bin/gunicorn", "app.server:app", \
    "--workers", "${WORKERS}", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:${PORT}", \
    "--log-level", "info"]