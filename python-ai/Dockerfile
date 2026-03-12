# Use NVIDIA CUDA base image with Python
# Using CUDA 12.1 with Ubuntu 22.04 and Python 3.10
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use the official Python index as primary and PyTorch as the extra source
# Force pip to use the global index for standard packages
# 1. Install standard dependencies first from PyPI
# Set the primary and extra indexes globally for the build
# Clear any previous index settings to prevent overrides
RUN pip config unset global.index-url || true
RUN pip config unset global.extra-index-url || true

# Run pip install with explicit priorities
# Step 1: Install standard web and database libraries from the global index ONLY
# 1. Install EVERY standard utility first from the main Python index
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    python-dotenv==1.0.0 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.25 \
    httpx==0.26.0 \
    python-multipart==0.0.6 \
    python-json-logger==2.0.7 \
    psutil==5.9.8

# 2. ONLY use the specialized index for the stripped-down requirements file
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# Download the small English model for spaCy
# Download the small English model for spaCy directly via pip
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

# Create necessary directories
RUN mkdir -p /app/logs /app/model_cache /app/app

# Copy application code
COPY app /app/app
COPY scripts /app/scripts

# Create non-root user for security
RUN useradd -m -u 1000 maritime && \
    chown -R maritime:maritime /app

# Switch to non-root user
USER maritime

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]