# Maritime/Trade Intelligence Agent

Production-ready, GPU-accelerated maritime intelligence platform using n8n, Python FastAPI, and PostgreSQL.

## 🚀 Features

- **GPU-Accelerated NLP**: NVIDIA RTX 3050 support for FinBERT and transformer models
- **Workflow Automation**: n8n for orchestrating complex intelligence workflows
- **Scalable API**: FastAPI with async support and automatic documentation
- **Production-Ready**: Docker Compose setup with health checks and persistent storage
- **Financial NLP**: FinBERT for sentiment analysis on maritime trade news

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 11 with WSL2 (Ubuntu 22.04 recommended)
- **GPU**: NVIDIA RTX 3050 (4-6GB VRAM)
- **RAM**: 16GB minimum
- **Storage**: 20GB free space

### Required Software
1. **Docker Desktop for Windows** (with WSL2 backend)
   - Download from: https://www.docker.com/products/docker-desktop/

2. **NVIDIA GPU Drivers** (Latest)
   - Download from: https://www.nvidia.com/download/index.aspx

3. **NVIDIA Container Toolkit** (in WSL2)
   ```bash
   # Run inside WSL2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Verify GPU Access**
   ```bash
   # In WSL2
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see your RTX 3050 listed.

## 🛠️ Installation & Setup

### Step 1: Clone/Download the Project

```bash
cd ~
# If you have this as a git repo:
git clone <your-repo-url> maritime-intel-agent
cd maritime-intel-agent

# Or create the structure manually (if you have the files)
```

### Step 2: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
```

**Required changes in `.env`:**
```bash
# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -hex 32)
N8N_BASIC_AUTH_PASSWORD=$(openssl rand -hex 16)
N8N_ENCRYPTION_KEY=$(openssl rand -hex 32)
PYTHON_AI_API_KEY=$(openssl rand -hex 32)
```

### Step 3: Create Required Directories

```bash
# The docker-compose will create these, but you can pre-create them:
mkdir -p n8n-data db-data/postgres logs/{n8n,python-ai,postgres}
```

### Step 4: Build and Start the Stack

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Or start services one by one
docker-compose up -d postgres
docker-compose up -d python-ai
docker-compose up -d n8n
```

### Step 5: Verify GPU Access

```bash
# Check Python AI service can see the GPU
docker-compose exec python-ai python /app/scripts/check_gpu.py

# Or use the API endpoint
curl http://localhost:8000/gpu-info
```

Expected output should show your RTX 3050.

### Step 6: Access the Services

- **n8n Workflow UI**: http://localhost:5678
  - Username: (from N8N_BASIC_AUTH_USER in .env)
  - Password: (from N8N_BASIC_AUTH_PASSWORD in .env)

- **Python AI API Docs**: http://localhost:8000/docs
  - Interactive Swagger documentation

- **Python AI GPU Info**: http://localhost:8000/gpu-info
  - GPU status and capabilities

- **PostgreSQL**: localhost:5432
  - Use any PostgreSQL client with credentials from .env

## 🧪 Testing the Setup

### 1. GPU Verification
```bash
# Run comprehensive GPU check
docker-compose exec python-ai python /app/scripts/check_gpu.py

# Test GPU computation
curl -X POST http://localhost:8000/api/v1/gpu/test \
  -H "Content-Type: application/json" \
  -d '{"matrix_size": 2000}'
```

### 2. API Health Check
```bash
# Basic health
curl http://localhost:8000/health

# System info
curl http://localhost:8000/system-info

# GPU detailed info
curl http://localhost:8000/gpu-info
```

### 3. Test Sentiment Analysis (Placeholder)
```bash
curl -X POST http://localhost:8000/api/v1/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The maritime shipping sector shows strong growth potential",
    "model": "finbert",
    "use_gpu": true
  }'
```

## 📁 Project Structure

```
maritime-intel-agent/
├── docker-compose.yml          # Main orchestration
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Template
├── .gitignore
├── README.md
│
├── python-ai/                 # Python FastAPI Service
│   ├── Dockerfile            # GPU-enabled container
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── api/routes.py    # API endpoints
│   │   ├── models/nlp.py    # FinBERT loader
│   │   └── utils/gpu_check.py
│   └── scripts/
│       └── check_gpu.py     # GPU verification
│
├── n8n-data/                 # n8n persistent data
├── db-data/                  # PostgreSQL data
└── logs/                     # Application logs
```

## 🔧 Common Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart a specific service
docker-compose restart python-ai

# View logs
docker-compose logs -f python-ai
docker-compose logs -f n8n

# Rebuild after code changes
docker-compose up -d --build python-ai

# Access container shell
docker-compose exec python-ai bash

# Check service status
docker-compose ps

# Remove everything (including volumes)
docker-compose down -v
```

## 🐛 Troubleshooting

### GPU Not Detected

1. **Verify NVIDIA drivers in WSL2:**
   ```bash
   nvidia-smi
   ```

2. **Check Docker GPU support:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Verify container GPU access:**
   ```bash
   docker-compose exec python-ai nvidia-smi
   ```

4. **Common fixes:**
   - Restart Docker Desktop
   - Restart WSL2: `wsl --shutdown` then reopen
   - Update NVIDIA drivers
   - Reinstall nvidia-container-toolkit

### Container Won't Start

```bash
# Check logs
docker-compose logs python-ai

# Check if port is already in use
netstat -ano | findstr :8000

# Remove and recreate
docker-compose down
docker-compose up -d --force-recreate
```

### Out of GPU Memory

```bash
# Clear GPU cache
curl -X POST http://localhost:8000/api/v1/gpu/clear-cache

# Check memory usage
curl http://localhost:8000/gpu-info

# Reduce batch size in your requests
```

### n8n Can't Connect to Python AI

1. Check network connectivity:
   ```bash
   docker-compose exec n8n ping python-ai
   ```

2. Verify Python AI is healthy:
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

3. Check n8n environment variables for correct URL

## 📚 Next Steps

1. **Implement FinBERT Model Loading**
   - Edit `python-ai/app/models/nlp.py`
   - Uncomment model loading in `python-ai/app/api/routes.py`

2. **Create n8n Workflows**
   - Access n8n UI at http://localhost:5678
   - Create workflows that call the Python AI API
   - Example: News scraping → Sentiment analysis → Database storage

3. **Add More Models**
   - Named Entity Recognition for ship/company names
   - Text classification for trade document types
   - Embeddings for semantic search

4. **Implement Database Schema**
   - Create tables for trade data
   - Add SQLAlchemy models
   - Implement data persistence

5. **Add Authentication**
   - API key validation
   - JWT tokens
   - Rate limiting

## 🔒 Security Considerations

- Change all default passwords in `.env`
- Never commit `.env` to version control
- Use HTTPS in production
- Implement proper API authentication
- Regular security updates
- Network isolation for production

## 📊 Performance Tips

- **Batch Processing**: Use batch endpoints for multiple texts
- **GPU Memory**: Monitor with `/gpu-info`, clear cache if needed
- **Model Caching**: Models are cached in Docker volume
- **Sequence Length**: Truncate long texts to 512 tokens
- **Concurrent Requests**: FastAPI handles async requests efficiently

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines]

## 📧 Support

[Add support contact or links]