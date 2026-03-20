# Dockerfile for Algo Bot
FROM python:3.10-slim

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

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Default environment variables
ENV PYTHONUNBUFFERED=1

# Use entrypoint to run both Agent + Bot
CMD ["./entrypoint.sh"]
