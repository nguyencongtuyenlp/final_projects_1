# 🤖 Multimodal AI Assistant

**Professional-grade multimodal AI assistant** với khả năng xử lý **text, image, và audio** trong một hệ thống thống nhất. Được thiết kế để production-ready với kiến trúc sạch, testing đầy đủ, và deployment dễ dàng.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Comprehensive-brightgreen.svg)](./tests/)

## ✨ **Tính năng chính**

### 🔤 **Text Processing**
- **Summarization**: Tóm tắt văn bản thông minh
- **Question Answering**: Trả lời câu hỏi dựa trên context
- **RAG (Retrieval-Augmented Generation)**: Tìm kiếm semantic + QA

### 👁️ **Vision Processing**  
- **OCR**: Trích xuất text từ hình ảnh (TrOCR)
- **VQA**: Trả lời câu hỏi về hình ảnh (BLIP)
- **Grad-CAM**: Giải thích kết quả phân loại ảnh

### 🎵 **Audio Processing**
- **ASR**: Chuyển giọng nói thành text (Wav2Vec2)
- **VAD**: Phát hiện hoạt động giọng nói
- **TTS**: Chuyển text thành giọng nói
- **Streaming**: Xử lý audio realtime qua WebSocket

### 🔥 **Real-Time Computer Vision** *(NEW!)*
- **Hand Gesture Recognition**: Nhận diện cử chỉ tay realtime (MediaPipe)
- **Eye Gaze Tracking**: Theo dõi hướng nhìn + phát hiện nháy mắt
- **Face Recognition**: Nhận diện khuôn mặt từ database (optional)
- **WebSocket Streaming**: Xử lý video frame realtime
- **Web Demo Interface**: Test trực tiếp qua browser

---

## 🏗️ **Kiến trúc hệ thống**

```
multimodal-assistant/
├── app/                          # 🚀 FastAPI Application
│   ├── pipelines/               # 🔧 AI Processing Pipelines
│   │   ├── nlp.py              # Text processing
│   │   ├── vision.py           # Image processing  
│   │   ├── audio.py            # Audio processing
│   │   ├── multimodal.py       # Multimodal orchestration
│   │   ├── streaming.py        # Realtime processing
│   │   └── retrieval.py        # Semantic search
│   ├── models/                  # 🤖 Model Management
│   │   └── registry.py         # Centralized model loading
│   ├── services/               # 🔗 Business Logic
│   │   ├── orchestrator.py     # Main coordinator
│   │   ├── rag_service.py      # RAG implementation
│   │   └── rag_store.py        # Vector storage
│   ├── utils/                  # 🛠️ Utilities
│   ├── xai/                    # 🔍 Explainable AI
│   ├── schemas.py              # 📋 API Schemas
│   └── server.py               # 🌐 FastAPI Server
├── train/                       # 🎓 Training Infrastructure
├── tests/                       # 🧪 Comprehensive Testing
├── docker/                      # 🐳 Containerization
└── docs/                        # 📚 Documentation
```

## 🚀 **Quick Start**

### **Phương pháp 1: Docker (Khuyên dùng)**
```bash
# Clone repository
git clone <your-repo-url>
cd multimodal-assistant

# Chạy với Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Hoặc development mode
docker-compose -f docker/docker-compose.dev.yml up --build
```

### **Phương pháp 2: Local Development**
```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Cài đặt dependencies
make dev-install

# Chạy server
make serve
```

### **Kiểm tra hoạt động**
```bash
# Health check
curl http://localhost:8000/health

# Swagger UI
open http://localhost:8000/docs

# Real-time demo
open http://localhost:8000/demo
```

### **Real-Time Dependencies**
```bash
# Essential (always install)
pip install mediapipe opencv-python

# Optional (for face recognition - requires CMake on Windows)
pip install face-recognition
```

---

## 📖 **Hướng dẫn sử dụng**

### **1. Multimodal Analysis**
```bash
# Phân tích text + image
curl -X POST "http://localhost:8000/v1/multimodal/analyze" \
  -F 'text=What is in this image?' \
  -F 'tasks=["ocr", "vqa"]' \
  -F 'image=@sample.jpg'
```

### **2. RAG Document Processing**
```bash
# Upload documents
curl -X POST "http://localhost:8000/v1/rag/upload" \
  -F "files=@document.pdf" \
  -F "files=@image.png"

# Query knowledge base  
curl -X POST "http://localhost:8000/v1/rag/query" \
  -d "question=What is the main topic?" \
  -d "top_k=5"
```

### **3. Audio Processing**
```bash
# Speech-to-text
curl -X POST "http://localhost:8000/v1/audio/asr" \
  -F "audio=@speech.wav"

# Text-to-speech
curl -X POST "http://localhost:8000/v1/audio/tts" \
  -d "text=Hello world" \
  --output speech.wav

# Voice activity detection
curl -X POST "http://localhost:8000/v1/audio/vad" \
  -F "audio=@audio.wav"
```

### **4. Explainable AI**
```bash
# Generate Grad-CAM visualization
curl -X POST "http://localhost:8000/v1/xai/gradcam" \
  -F "image=@image.jpg" \
  --output gradcam.png
```

### **5. Real-Time Computer Vision** 🔥
```bash
# Open web demo interface
open http://localhost:8000/demo

# WebSocket endpoints:
# - ws://localhost:8000/v1/realtime/gesture (Hand gestures)
# - ws://localhost:8000/v1/realtime/eyetracking (Eye gaze)
# - ws://localhost:8000/v1/realtime/vision (Combined)
```

**Web Demo Features:**
- 📹 **Live camera feed** with real-time processing
- 🖐️ **Hand gestures**: Fist, Thumbs Up, Peace, Open Palm
- 👁️ **Eye tracking**: Gaze direction + blink detection
- 📊 **Live statistics** and confidence scores
- 🎯 **Processed images** with landmark visualization

### **6. Realtime WebSocket**
```python
import asyncio
import websockets

async def stream_audio():
    uri = "ws://localhost:8000/v1/realtime/audio"
    async with websockets.connect(uri) as ws:
        # Send audio chunks
        await ws.send(audio_bytes)
        
        # Receive transcription
        result = await ws.recv()
        print(result)
```

**📋 Xem [API_DOCS.md](./API_DOCS.md) để biết chi tiết đầy đủ về các endpoints.**

## 🧪 **Development & Testing**

### **Testing**
```bash
# Chạy toàn bộ tests
make test

# Test với coverage report
make test-cov

# Test specific module
pytest tests/test_pipelines.py -v

# Test với watch mode (tự động re-run khi code thay đổi)
make test-watch
```

### **Code Quality**
```bash
# Format code
make fmt

# Type checking
mypy app/

# Security scan
bandit -r app/
```

### **Development Commands**
```bash
# Xem tất cả commands
make help

# Clean cache files
make clean

# Install development dependencies
make dev-install
```

---

## 🐳 **Deployment**

### **Production với Docker**
```bash
# Build production image
docker build -f docker/Dockerfile -t multimodal-assistant .

# Run production container
docker run -p 8000:8000 \
  -v $(pwd)/storage:/app/storage \
  -e APP_ENV=production \
  multimodal-assistant
```

### **Environment Variables**
```bash
# Core settings
APP_ENV=production          # development/production
APP_HOST=0.0.0.0           # Host to bind
APP_PORT=8000              # Port to listen

# Optional authentication
APP_AUTH_TOKEN=your_secret_token

# Model caching
TRANSFORMERS_CACHE=/path/to/model/cache
```

### **Health Monitoring**
```bash
# Health check endpoint
curl http://localhost:8000/health

# Metrics (planning)
curl http://localhost:8000/metrics
```

---

## 🤖 **Models & AI Components**

### **Pre-trained Models**
| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| OCR | microsoft/trocr-base-printed | ~558MB | Text extraction |
| VQA | Salesforce/blip-vqa-base | ~990MB | Visual Q&A |
| ASR | facebook/wav2vec2-base-960h | ~360MB | Speech-to-text |
| Summarization | sshleifer/distilbart-cnn-12-6 | ~306MB | Text summarization |
| QA | deepset/bert-base-cased-squad2 | ~436MB | Question answering |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | ~90MB | Semantic search |

### **Model Loading**
- Models tự động download khi chạy lần đầu
- Cached locally để sử dụng tiếp theo
- Sử dụng `@lru_cache` để memory efficiency

---

## 🔮 **Roadmap**

### **Phase 1 (Completed) ✅**
- [x] Multimodal API endpoints
- [x] RAG document processing  
- [x] Audio processing (ASR, TTS, VAD)
- [x] Realtime WebSocket streaming
- [x] Comprehensive testing
- [x] Docker deployment

### **Phase 2 (Planning) 🚧**
- [ ] Model fine-tuning infrastructure
- [ ] Batch processing endpoints
- [ ] Advanced caching (Redis)
- [ ] Performance optimization
- [ ] Monitoring & metrics

### **Phase 3 (Future) 🌟**  
- [ ] Multi-language support
- [ ] Custom model training UI
- [ ] Advanced XAI features
- [ ] Kubernetes deployment
- [ ] Auto-scaling capabilities

---

## 🤝 **Contributing**

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### **Development Setup**
```bash
# Clone và setup
git clone <repo-url>
cd multimodal-assistant
make dev-install

# Pre-commit hooks
pre-commit install

# Run tests trước khi commit
make test
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- [Hugging Face](https://huggingface.co/) cho pre-trained models
- [FastAPI](https://fastapi.tiangolo.com/) cho web framework
- [PyTorch](https://pytorch.org/) cho deep learning infrastructure.
