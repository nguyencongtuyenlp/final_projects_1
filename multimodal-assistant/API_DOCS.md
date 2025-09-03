# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Optional Bearer token authentication. Set `APP_AUTH_TOKEN` environment variable to enable.

```bash
Authorization: Bearer your_token_here
```

## Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
    "ok": true,
    "status": "healthy",
    "version": "1.0.0"
}
```

---

### Multimodal Analysis
Analyze text and/or image with specified tasks.

```http
POST /v1/multimodal/analyze
Content-Type: multipart/form-data
```

**Parameters:**
- `text` (optional): Text to analyze
- `image` (optional): Image file (JPEG, PNG, etc.)
- `tasks` (array): List of tasks to perform

**Tasks:**
- `ocr`: Extract text from image
- `vqa`: Visual question answering
- `summary`: Summarize text
- `qa`: Question answering (requires context)

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/multimodal/analyze" \
  -F 'text=What is in this image?' \
  -F 'tasks=["ocr", "vqa"]' \
  -F 'image=@image.jpg'
```

**Response:**
```json
{
    "ok": true,
    "result": {
        "ocr": "extracted text from image",
        "vqa": {
            "answer": "description of image content"
        }
    }
}
```

---

### RAG (Retrieval-Augmented Generation)

#### Upload Documents
```http
POST /v1/rag/upload
Content-Type: multipart/form-data
```

**Parameters:**
- `files`: Array of files (PDF, images, text files)

**Supported formats:**
- PDF documents
- Images (for OCR): JPEG, PNG, BMP, WebP
- Text files: TXT, markdown, etc.

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/rag/upload" \
  -F "files=@document.pdf" \
  -F "files=@image.png"
```

**Response:**
```json
{
    "ok": true,
    "chunks_added": 15
}
```

#### Query RAG
```http
POST /v1/rag/query
Content-Type: application/x-www-form-urlencoded
```

**Parameters:**
- `question` (required): Question to ask
- `top_k` (optional, default=5): Number of relevant chunks to retrieve

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/rag/query" \
  -d "question=What is the main topic of the document?" \
  -d "top_k=3"
```

**Response:**
```json
{
    "ok": true,
    "result": {
        "answer": {
            "answer": "The main topic is...",
            "start": 15,
            "end": 25
        },
        "hits": [
            {
                "text": "relevant chunk text",
                "score": 0.95,
                "rank": 0,
                "meta": {"filename": "document.pdf"}
            }
        ]
    }
}
```

---

### Audio Processing

#### Speech-to-Text (ASR)
```http
POST /v1/audio/asr
Content-Type: multipart/form-data
```

**Parameters:**
- `audio`: Audio file (WAV, MP3, etc.)

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/audio/asr" \
  -F "audio=@speech.wav"
```

**Response:**
```json
{
    "ok": true,
    "text": "transcribed speech text"
}
```

#### Voice Activity Detection (VAD)
```http
POST /v1/audio/vad
Content-Type: multipart/form-data
```

**Parameters:**
- `audio`: Audio file

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/audio/vad" \
  -F "audio=@audio.wav"
```

**Response:**
```json
{
    "ok": true,
    "segments": [
        [0.5, 2.1],
        [3.2, 5.8]
    ]
}
```

#### Text-to-Speech (TTS)
```http
POST /v1/audio/tts
Content-Type: application/x-www-form-urlencoded
```

**Parameters:**
- `text`: Text to synthesize (max 500 characters)

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/audio/tts" \
  -d "text=Hello world" \
  --output speech.wav
```

**Response:** WAV audio file

---

### Explainable AI (XAI)

#### Grad-CAM Visualization
```http
POST /v1/xai/gradcam
Content-Type: multipart/form-data
```

**Parameters:**
- `image`: Image file for explanation
- `class_idx` (optional): Target class index for explanation

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/xai/gradcam" \
  -F "image=@image.jpg" \
  -F "class_idx=1" \
  --output gradcam.png
```

**Response:** PNG image with Grad-CAM overlay

---

### WebSocket Endpoints

#### Realtime Audio Stream
```
ws://localhost:8000/v1/realtime/audio
```

**Protocol:**
- Send binary audio chunks
- Send text commands: `flush`, `reset`
- Receive JSON responses with transcriptions

**Example:**
```python
import websockets
import asyncio

async def stream_audio():
    uri = "ws://localhost:8000/v1/realtime/audio"
    async with websockets.connect(uri) as websocket:
        # Send audio chunks
        await websocket.send(audio_bytes)
        
        # Receive transcription
        response = await websocket.recv()
        print(response)
        
        # Flush remaining audio
        await websocket.send("flush")
```

#### Realtime TTS Stream
```
ws://localhost:8000/v1/realtime/tts
```

**Protocol:**
- Send text messages
- Receive audio bytes
- Send `__end__` to close

---

## Error Handling

All endpoints return consistent error format:

```json
{
    "ok": false,
    "error": "Error message",
    "detail": "Detailed error information (in debug mode)"
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (validation error)
- `401`: Unauthorized
- `413`: Request Entity Too Large
- `422`: Unprocessable Entity (Pydantic validation error)
- `500`: Internal Server Error

## Rate Limits
Currently no rate limits are enforced. Consider implementing rate limiting for production use.

## Model Information

**Pre-trained models used:**
- **OCR**: microsoft/trocr-base-printed
- **VQA**: Salesforce/blip-vqa-base
- **ASR**: facebook/wav2vec2-base-960h
- **Summarization**: sshleifer/distilbart-cnn-12-6
- **QA**: deepset/bert-base-cased-squad2
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Classification**: ResNet18 (for Grad-CAM)

Models are automatically downloaded on first use and cached locally.
