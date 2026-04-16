# Dockerfile for Algo Bot
FROM python:3.11-slim

# Install system dependencies for TA-Lib and other Python extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    sqlite3 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (C library)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Use uv run for better dependency awareness
ENV UV_COMPILE_BYTECODE=1

# Default environment variables
ENV PYTHONUNBUFFERED=1

# Use entrypoint to run both Agent + Bot
CMD ["./entrypoint.sh"]
