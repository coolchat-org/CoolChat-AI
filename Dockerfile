# # Sử dụng Ubuntu làm base image
# FROM ubuntu:22.04

# # Thiết lập biến môi trường
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Asia/Ho_Chi_Minh
# # ENV PATH="$HOME/.local/bin:$PATH"
# ENV PATH="/root/.local/bin:$PATH"

# # Cài đặt các gói cần thiết
# RUN apt-get update && apt-get install -y \
#     wget \
#     curl \
#     tzdata \
#     libreoffice \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Cài đặt UV Package Manager
# RUN wget -qO- https://astral.sh/uv/install.sh | sh

# # Thiết lập thư mục làm việc
# WORKDIR /app

# # Sao chép mã nguồn vào container
# COPY . .

# # Đồng bộ dependencies qua UV
# RUN uv sync

# # Expose cổng ứng dụng
# EXPOSE 8000

# # Lệnh khởi động ứng dụng
# CMD [".venv/bin/uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Sử dụng Ubuntu làm base image
FROM ubuntu:22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV PATH="/root/.local/bin:$PATH"

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tzdata \
    libreoffice \
    # Các gói phụ thuộc cần thiết cho Playwright
    python3 \
    python3-pip \
    # Các dependencies cho trình duyệt của Playwright
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
    # Các gói bổ sung cho các trình duyệt khác nhau
    libxcursor1 \
    libxtst6 \
    libxshmfence-dev \
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

# Cài đặt Playwright và browser
RUN .venv/bin/playwright install --with-deps chromium

# Expose cổng ứng dụng
EXPOSE 8000

# Lệnh khởi động ứng dụng
CMD [".venv/bin/uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]