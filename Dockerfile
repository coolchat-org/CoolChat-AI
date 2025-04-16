FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Ho_Chi_Minh \
    PATH="/root/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
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
    libz-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install grpcio==1.60.1 && \
    .venv/bin/pip install -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD [".venv/bin/gunicorn", "app.server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]