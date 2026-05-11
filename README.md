# Alpaca 🦙
### Standalone Inference & Model Management Stack

Alpaca is a high-performance LLM inference stack designed to provide Ollama-compatible API parity while bypassing the official Ollama service. It leverages the `llama-cpp-turboquant` engine for state-of-the-art inference and a custom proxy for dynamic model lifecycle management.

---

## 🏗️ Architecture

- **Alpaca Engine**: A containerized `llama-server` built with TurboQuant support.
- **Alpaca Proxy**: A FastAPI-based bridge that mimics the Ollama API (`/api/chat`, `/api/tags`, `/api/ps`, etc.) and orchestrates model switching by updating the engine configuration and restarting the container.
- **Alpaca Puller**: A standalone Python utility to download and register models directly from the Ollama Registry or Hugging Face (via manual import).

---

## 🚀 Getting Started

### 1. Prerequisites
- Docker & Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- Python 3.10+

### 2. Deployment
Bring up the entire stack using Docker Compose:
```bash
cd ~/llama-test
sudo docker compose up -d --build
```

The stack will be available at `http://localhost:11434`, perfectly mimicking an Ollama instance.

---

## 🛠️ Components

### Alpaca Proxy (`alpaca-proxy.py`)
The heart of the stack. It listens on port `11434` and handles:
- **Model Switching**: Automatically updates `docker-compose.yml` and restarts the engine when a different model is requested.
- **Client Stability**: Sends heartbeats to the client during model loads to prevent timeouts in UIs like OpenWebUI.
- **Local Discovery**: Scans the manifests directory to provide a real-time list of available models.

### Alpaca Puller (`alpaca-puller.py`)
Download models without the official Ollama service:
```bash
# Pull a new model
sudo python3 alpaca-puller.py pull llama3:8b

# Remove a model
sudo python3 alpaca-puller.py remove llama3:8b
```

---

## 🧪 Testing

Run the integrated diagnostic suite to verify model switching and API health:
```bash
python3 test-alpaca.py
```

---

## 📦 Directory Structure
- `/models`: Mounted to your host's Ollama models directory.
- `alpaca-proxy.py`: The management bridge.
- `alpaca-puller.py`: The standalone model downloader.
- `docker-compose.yml`: Defines the service orchestration.
