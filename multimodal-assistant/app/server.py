from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
import json
from PIL import Image
import io
import traceback
import json
from app.services.orchestrator import Orchestrator
from app.schemas import *
from app.pipelines.realtime_vision import RealTimeVisionPipeline

app = FastAPI(
    title="Multimodal AI Assistant",
    version="1.0.0",
    description="Professional multimodal AI assistant with vision, audio, and NLP capabilities",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_detail = str(exc)
    if app.debug:
        error_detail = traceback.format_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"ok": False, "error": "Internal server error", "detail": error_detail}
    )

# Initialize services
orc = Orchestrator()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"ok": True, "status": "healthy", "version": "1.0.0"}


@app.get("/demo")
async def real_time_demo():
    """Serve real-time vision demo page"""
    return FileResponse("static/realtime_demo.html")

@app.post("/v1/multimodal/analyze", response_model=AnalyzeResponse)
async def analyze_multimodal(
    text: Optional[str] = Form(default=None),
    tasks: List[str] = Form(default=[]),
    image: Optional[UploadFile] = File(default=None),
):
    """
    Analyze multimodal input (text + image) with specified tasks
    
    Tasks:
    - ocr: Extract text from image
    - vqa: Visual question answering  
    - summary: Summarize text
    - qa: Question answering (requires context)
    """
    try:
        # Validate file type if image provided
        if image is not None:
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be an image"
                )
            content = await image.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
        else:
            img = None

        # Normalize tasks: handle case when client sends JSON string like '["ocr"]'
        norm_tasks = tasks
        if isinstance(norm_tasks, list) and len(norm_tasks) == 1 and isinstance(norm_tasks[0], str):
            candidate = norm_tasks[0].strip()
            if (candidate.startswith('[') and candidate.endswith(']')) or (candidate.startswith('"') and candidate.endswith('"')):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        norm_tasks = parsed
                except Exception:
                    pass

        # Normalize casing/whitespace
        norm_tasks = [t.strip().lower() for t in (norm_tasks or []) if isinstance(t, str) and t.strip()]

        # Validate request
        request_data = AnalyzeRequest(text=text, tasks=norm_tasks)
        
        result = orc.analyze(text=text, image=img, tasks=request_data.tasks)
        return AnalyzeResponse(result=result)
        
    except HTTPException:
        raise
    except Exception as e:
        return AnalyzeResponse(ok=False, error=str(e))


# ============ RAG ENDPOINTS ============
from app.services.rag_service import RAGService

rag = RAGService()

@app.post("/v1/rag/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents to RAG store (supports PDF, images, text files)
    """
    try:
        total_chunks = 0
        for f in files:
            # Validate file size (10MB limit)
            content = await f.read()
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {f.filename} too large (max 10MB)"
                )
            
            meta = {"filename": f.filename, "content_type": f.content_type}
            text = ""
            
            if f.content_type == "application/pdf" or f.filename.lower().endswith(".pdf"):
                from pdfminer.high_level import extract_text
                try:
                    text = extract_text(io.BytesIO(content))
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to extract text from PDF: {str(e)}"
                    )
            elif f.content_type.startswith("image/"):
                from app.pipelines.vision import VisionPipeline
                try:
                    img = Image.open(io.BytesIO(content)).convert("RGB")
                    vp = VisionPipeline()
                    text = vp.ocr(img)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to process image: {str(e)}"
                    )
            else:
                # Assume text file
                try:
                    text = content.decode("utf-8", errors="ignore")
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to decode text: {str(e)}"
                    )

            if text.strip():
                total_chunks += rag.add_texts([text], meta=meta)

        return {"ok": True, "chunks_added": total_chunks}
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ok": False, "error": str(e)}
        )

@app.post("/v1/rag/query", response_model=RAGQueryResponse)
async def query_rag(
    question: str = Form(..., description="Question to ask"),
    top_k: int = Form(default=5, description="Number of top results to retrieve")
):
    """
    Query RAG store with semantic search + QA
    """
    try:
        request_data = RAGQueryRequest(question=question, top_k=top_k)
        result = rag.query(request_data.question, top_k=request_data.top_k)
        return RAGQueryResponse(result=result)
        
    except Exception as e:
        return RAGQueryResponse(ok=False, error=str(e))

# ============ XAI ENDPOINTS ============
from app.xai.gradcam import GradCAM, overlay_heatmap

@app.post("/v1/xai/gradcam")
async def generate_gradcam(
    image: UploadFile = File(..., description="Image for explanation"),
    class_idx: Optional[int] = Form(default=None, description="Target class index")
):
    """
    Generate Grad-CAM visualization for image classification
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
            
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        g = GradCAM(target_layer="layer4")
        heatmap, predicted_class = g.generate(img, class_idx=class_idx)
        overlay = overlay_heatmap(img, heatmap, alpha=0.45)
        
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate Grad-CAM: {str(e)}"
        )

# ============ AUDIO ENDPOINTS ============
from app.pipelines.audio import AudioPipeline

audio_pipeline = AudioPipeline()

@app.post("/v1/audio/asr", response_model=ASRResponse)
async def speech_to_text(audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")):
    """
    Convert speech to text using Wav2Vec2
    """
    try:
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an audio file"
            )
            
        content = await audio.read()
        result = audio_pipeline.asr(content)
        return ASRResponse(text=result.get("text", ""))
        
    except HTTPException:
        raise
    except Exception as e:
        return ASRResponse(ok=False, error=str(e))

@app.post("/v1/audio/vad", response_model=VADResponse)
async def voice_activity_detection(audio: UploadFile = File(..., description="Audio file")):
    """
    Detect voice activity segments in audio
    """
    try:
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an audio file"
            )
            
        content = await audio.read()
        result = audio_pipeline.vad_segments(content, sample_rate=16000)
        return VADResponse(segments=result.get("segments", []))
        
    except HTTPException:
        raise
    except Exception as e:
        return VADResponse(ok=False, error=str(e))

@app.post("/v1/audio/tts")
async def text_to_speech(text: str = Form(..., description="Text to synthesize")):
    """
    Convert text to speech
    """
    try:
        request_data = TTSRequest(text=text)
        audio_bytes = audio_pipeline.tts(request_data.text)
        
        return StreamingResponse(
            io.BytesIO(audio_bytes), 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate speech: {str(e)}"
        )


# ============ WEBSOCKET ENDPOINTS ============
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from app.pipelines.streaming import StreamingASR
import asyncio

@app.websocket("/v1/realtime/audio")
async def realtime_audio_stream(websocket: WebSocket):
    """
    Realtime audio streaming with ASR
    
    Send binary audio chunks and receive transcription results.
    Send text commands: 'flush', 'reset'
    """
    await websocket.accept()
    stream = StreamingASR()
    
    try:
        await websocket.send_json({
            "type": "info", 
            "message": "Connected to realtime audio stream",
            "commands": ["flush", "reset"]
        })
        
        while True:
            msg = await websocket.receive()
            
            if "bytes" in msg and msg["bytes"] is not None:
                # Binary audio chunk
                try:
                    results = stream.push_bytes(msg["bytes"])
                    for result in results:
                        await websocket.send_json(result)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"ASR processing error: {str(e)}"
                    })
                    
            elif "text" in msg and msg["text"] is not None:
                # Text command
                cmd = msg["text"].strip().lower()
                
                if cmd == "flush":
                    try:
                        result = stream.flush()
                        if result:
                            await websocket.send_json(result)
                        else:
                            await websocket.send_json({
                                "type": "info",
                                "message": "No audio to flush"
                            })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Flush error: {str(e)}"
                        })
                        
                elif cmd == "reset":
                    try:
                        stream.reset()
                        await websocket.send_json({
                            "type": "info",
                            "message": "Stream reset successfully"
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Reset error: {str(e)}"
                        })
                        
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown command: {cmd}. Available: flush, reset"
                    })
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/v1/realtime/tts")
async def realtime_tts_stream(websocket: WebSocket):
    """
    Realtime text-to-speech streaming
    
    Send text messages and receive audio chunks.
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "info",
            "message": "Connected to realtime TTS stream"
        })
        
        while True:
            msg = await websocket.receive_text()
            
            if msg.strip().lower() == "__end__":
                await websocket.send_json({
                    "type": "info",
                    "message": "TTS stream ended"
                })
                break
                
            try:
                # Generate audio in background thread
                audio_bytes = await run_in_threadpool(
                    lambda: audio_pipeline.tts(msg)
                )
                await websocket.send_bytes(audio_bytes)
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"TTS error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error", 
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# Real-Time Vision WebSocket Endpoints
rt_vision = RealTimeVisionPipeline()

@app.websocket("/v1/realtime/gesture")
async def websocket_gesture_recognition(websocket: WebSocket):
    """
    Real-time hand gesture recognition via WebSocket
    
    Expected message format:
    {
        "type": "image",
        "data": "base64_encoded_image"
    }
    
    Response format:
    {
        "type": "gesture_recognition",
        "hand_count": int,
        "gestures": [{"hand": "Left/Right", "gesture": "Fist", "confidence": 0.85}],
        "dominant_gesture": "Fist",
        "processed_image": "base64_encoded_image_with_landmarks",
        "timestamp": float
    }
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "🖐️ Hand Gesture Recognition WebSocket connected!",
            "instructions": [
                "Send base64 encoded images",
                "Show hand gestures: Fist, Thumbs Up, Peace, Open Palm",
                "Real-time landmark detection and classification"
            ]
        })
        
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                if data.get("type") == "image":
                    # Process gesture recognition
                    result = rt_vision.process_gesture_recognition(data.get("data", ""))
                    await websocket.send_json(result)
                    
                elif data.get("type") == "reset":
                    # Reset session
                    result = rt_vision.reset_session()
                    await websocket.send_json(result)
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type. Use 'image' or 'reset'"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/v1/realtime/eyetracking")
async def websocket_eye_tracking(websocket: WebSocket):
    """
    Real-time eye gaze tracking and blink detection via WebSocket
    
    Expected message format:
    {
        "type": "image", 
        "data": "base64_encoded_image"
    }
    
    Response format:
    {
        "type": "eye_tracking",
        "gaze_data": {
            "left_gaze": "Left/Right/Center",
            "right_gaze": "Left/Right/Center", 
            "blink_detected": bool,
            "ear_left": float,
            "ear_right": float,
            "total_blinks": int
        },
        "processed_image": "base64_encoded_image_with_landmarks",
        "timestamp": float
    }
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "👁️ Eye Gaze Tracking WebSocket connected!",
            "instructions": [
                "Send base64 encoded images of your face",
                "Look left, right, center to test gaze tracking",
                "Blink normally to test blink detection",
                "Ensure good lighting on your face"
            ]
        })
        
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                if data.get("type") == "image":
                    # Process eye tracking
                    result = rt_vision.process_eye_tracking(data.get("data", ""))
                    await websocket.send_json(result)
                    
                elif data.get("type") == "reset":
                    # Reset session
                    result = rt_vision.reset_session()
                    await websocket.send_json(result)
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type. Use 'image' or 'reset'"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/v1/realtime/vision")
async def websocket_combined_vision(websocket: WebSocket):
    """
    Combined real-time vision processing (gestures + eye tracking)
    
    Expected message format:
    {
        "type": "image",
        "data": "base64_encoded_image",
        "features": ["gestures", "eye_tracking"]  // optional, defaults to both
    }
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "🔥 Combined Vision WebSocket connected!",
            "features": ["Hand Gesture Recognition", "Eye Gaze Tracking"],
            "instructions": [
                "Send base64 encoded images",
                "Optionally specify features: ['gestures', 'eye_tracking']",
                "Default: processes both features simultaneously"
            ]
        })
        
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                if data.get("type") == "image":
                    base64_image = data.get("data", "")
                    features = data.get("features", ["gestures", "eye_tracking"])
                    
                    results = {
                        "type": "combined_vision",
                        "timestamp": rt_vision.session_start,
                        "features_processed": features
                    }
                    
                    # Process gesture recognition
                    if "gestures" in features:
                        gesture_result = rt_vision.process_gesture_recognition(base64_image)
                        results["gesture_recognition"] = gesture_result
                    
                    # Process eye tracking  
                    if "eye_tracking" in features:
                        eye_result = rt_vision.process_eye_tracking(base64_image)
                        results["eye_tracking"] = eye_result
                    
                    await websocket.send_json(results)
                    
                elif data.get("type") == "reset":
                    # Reset session
                    result = rt_vision.reset_session()
                    await websocket.send_json(result)
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type. Use 'image' or 'reset'"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
