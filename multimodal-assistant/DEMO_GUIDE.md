# 🎮 Demo Guide - Multimodal AI Assistant

Hướng dẫn test và demo các tính năng của hệ thống một cách nhanh chóng.

## 🚀 Quick Start Demo

### 1. **Khởi động hệ thống**
```bash
# Option 1: Docker (khuyên dùng)
docker-compose -f docker/docker-compose.yml up --build

# Option 2: Local development
make dev-install
make serve
```

### 2. **Kiểm tra hệ thống hoạt động**
```bash
# Health check
curl http://localhost:8000/health

# Mở Swagger UI để explore APIs
open http://localhost:8000/docs
```

### 3. **Run Interactive Demo**
```bash
# Interactive demo với menu
python demo.py

# Hoặc chạy full demo
python demo.py --full
```

---

## 📖 **Manual Testing Guide**

### **🏥 Health Check**
```bash
curl -X GET "http://localhost:8000/health"
```

**Expected Response:**
```json
{
    "ok": true,
    "status": "healthy",
    "version": "1.0.0"
}
```

---

### **🤖 Multimodal Analysis**

#### **OCR (Text Extraction from Image)**
```bash
# Download sample image hoặc sử dụng ảnh của bạn
curl -X POST "http://localhost:8000/v1/multimodal/analyze" \
  -F 'tasks=["ocr"]' \
  -F 'image=@sample_image.jpg'
```

#### **Text Summarization**
```bash
curl -X POST "http://localhost:8000/v1/multimodal/analyze" \
  -F 'text=Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.' \
  -F 'tasks=["summary"]'
```

#### **Visual Question Answering (VQA)**
```bash
curl -X POST "http://localhost:8000/v1/multimodal/analyze" \
  -F 'text=What objects are in this image?' \
  -F 'tasks=["vqa"]' \
  -F 'image=@sample_image.jpg'
```

---

### **📚 RAG (Document Processing)**

#### **Upload Documents**
```bash
# Upload PDF
curl -X POST "http://localhost:8000/v1/rag/upload" \
  -F "files=@sample_document.pdf"

# Upload image for OCR
curl -X POST "http://localhost:8000/v1/rag/upload" \
  -F "files=@document_image.png"

# Upload text file
curl -X POST "http://localhost:8000/v1/rag/upload" \
  -F "files=@sample_text.txt"
```

#### **Query Knowledge Base**
```bash
curl -X POST "http://localhost:8000/v1/rag/query" \
  -d "question=What is the main topic of the document?" \
  -d "top_k=5"
```

---

### **🎵 Audio Processing**

#### **Speech-to-Text (ASR)**
```bash
curl -X POST "http://localhost:8000/v1/audio/asr" \
  -F "audio=@sample_speech.wav"
```

#### **Text-to-Speech (TTS)**
```bash
curl -X POST "http://localhost:8000/v1/audio/tts" \
  -d "text=Hello from multimodal AI assistant!" \
  --output generated_speech.wav
```

#### **Voice Activity Detection (VAD)**
```bash
curl -X POST "http://localhost:8000/v1/audio/vad" \
  -F "audio=@audio_with_speech.wav"
```

---

### **🔍 Explainable AI (XAI)**

#### **Grad-CAM Visualization**
```bash
curl -X POST "http://localhost:8000/v1/xai/gradcam" \
  -F "image=@sample_image.jpg" \
  --output gradcam_visualization.png
```

---

### **🌐 WebSocket Demo**

#### **Realtime Audio Streaming**
```python
import asyncio
import websockets

async def test_realtime_audio():
    uri = "ws://localhost:8000/v1/realtime/audio"
    async with websockets.connect(uri) as websocket:
        # Send audio file in chunks
        with open("sample_audio.wav", "rb") as f:
            chunk_size = 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                await websocket.send(chunk)
        
        # Flush and get final result
        await websocket.send("flush")
        
        # Receive transcription results
        async for message in websocket:
            print(f"Received: {message}")
            break

asyncio.run(test_realtime_audio())
```

#### **Realtime TTS Streaming**
```python
import asyncio
import websockets

async def test_realtime_tts():
    uri = "ws://localhost:8000/v1/realtime/tts"
    async with websockets.connect(uri) as websocket:
        # Send text for synthesis
        await websocket.send("Hello from realtime TTS!")
        
        # Receive audio data
        audio_data = await websocket.recv()
        
        # Save audio
        with open("realtime_tts_output.wav", "wb") as f:
            f.write(audio_data)
        
        # End session
        await websocket.send("__end__")

asyncio.run(test_realtime_tts())
```

---

## 📊 **Performance Testing**

### **Load Testing với curl**
```bash
# Test concurrent requests
for i in {1..10}; do
  curl -X GET "http://localhost:8000/health" &
done
wait
```

### **Benchmark với Apache Bench**
```bash
# Install ab first: sudo apt-get install apache2-utils
ab -n 100 -c 10 http://localhost:8000/health
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Server không khởi động được**
```bash
# Check logs
docker-compose logs -f

# Check port conflicts
lsof -i :8000
```

#### **2. Model loading errors**
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Check disk space
df -h
```

#### **3. Memory issues**
```bash
# Monitor memory usage
docker stats

# Increase Docker memory limit nếu cần
```

#### **4. Audio processing errors**
```bash
# Install system audio dependencies (Ubuntu)
sudo apt-get install libsndfile1 espeak-ng ffmpeg

# Check audio file format
file sample_audio.wav
```

---

## 📈 **Expected Results**

### **Successful Responses**
- **Health Check**: `{"ok": true, "status": "healthy"}`
- **OCR**: Text extracted from image
- **Summarization**: Shortened version of input text
- **VQA**: Descriptive answer about image content
- **ASR**: Transcribed text from audio
- **TTS**: Generated WAV audio file
- **Grad-CAM**: PNG image with heatmap overlay

### **Performance Benchmarks**
- **Health Check**: < 50ms
- **OCR**: 1-3 seconds (depending on image size)
- **Summarization**: 0.5-2 seconds
- **ASR**: 1-5 seconds (depending on audio length)
- **TTS**: 0.5-2 seconds
- **RAG Query**: 1-3 seconds

---

## 📝 **Sample Files**

Tạo sample files để test:

### **Sample Text File (sample.txt)**
```
Multimodal AI Assistant là một hệ thống AI tiên tiến.
Hệ thống có khả năng xử lý text, image và audio.
Được xây dựng với PyTorch, FastAPI và Docker.
```

### **Sample Image Script**
```python
from PIL import Image, ImageDraw
img = Image.new('RGB', (400, 200), 'white')
draw = ImageDraw.Draw(img)
draw.text((50, 50), "Test OCR Image\nMultimodal AI", fill='black')
img.save('sample_image.jpg')
```

### **Sample Audio Script**
```python
import numpy as np
import wave

# Generate 2-second sine wave
duration = 2.0
sample_rate = 16000
frequency = 440
t = np.linspace(0, duration, int(sample_rate * duration), False)
wave_data = np.sin(frequency * 2 * np.pi * t)
wave_data = (wave_data * 32767).astype(np.int16)

with wave.open('sample_audio.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(wave_data.tobytes())
```

---

## 🎯 **Next Steps**

1. **Explore Swagger UI** at http://localhost:8000/docs
2. **Read API Documentation** in [API_DOCS.md](./API_DOCS.md)
3. **Check source code** để hiểu implementation
4. **Contribute** với new features hoặc improvements
5. **Deploy to production** với Docker

**Happy Testing! 🚀**
