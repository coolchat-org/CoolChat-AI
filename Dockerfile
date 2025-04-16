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

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv pip install grpcio==1.60.1 --config-settings="--global-option=build_ext" --config-settings="--global-option=-I/usr/include/openssl" && \
    uv sync --frozen

# Copy application code
COPY . .

EXPOSE 8000

CMD [".venv/bin/gunicorn", "app.server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]