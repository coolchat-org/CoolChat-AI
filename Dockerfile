FROM python:3.11-slim

# Metadata
LABEL maintainer="Your Name"
LABEL version="1.0"
LABEL description="FastAPI LLM Application"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV WORKERS=4
ENV TZ=Asia/Ho_Chi_Minh

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install UV Package Manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE ${PORT}


# Start application
CMD [".venv/bin/gunicorn", "app.server:app", \
    "--workers", "${WORKERS}", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:${PORT}", \
    "--log-level", "info"]