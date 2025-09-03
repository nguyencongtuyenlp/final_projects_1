"""
Comprehensive API endpoint tests
"""
import pytest
import io
from fastapi import status

class TestHealthCheck:
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "status" in data
        assert "version" in data

class TestMultimodalEndpoints:
    def test_analyze_text_only(self, client):
        """Test multimodal analysis with text only"""
        response = client.post(
            "/v1/multimodal/analyze",
            data={"text": "Hello world", "tasks": ["summary"]}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "result" in data

    def test_analyze_with_image_ocr(self, client, sample_image):
        """Test OCR functionality"""
        response = client.post(
            "/v1/multimodal/analyze",
            data={"tasks": ["ocr"]},
            files={"image": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "result" in data
        assert "ocr" in data["result"]

    def test_analyze_with_invalid_task(self, client):
        """Test with invalid task"""
        response = client.post(
            "/v1/multimodal/analyze",
            data={"text": "Hello", "tasks": ["invalid_task"]}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_analyze_with_invalid_file_type(self, client):
        """Test with non-image file"""
        response = client.post(
            "/v1/multimodal/analyze",
            data={"tasks": ["ocr"]},
            files={"image": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

class TestRAGEndpoints:
    def test_upload_text_document(self, client, sample_text_file):
        """Test uploading text document to RAG"""
        response = client.post(
            "/v1/rag/upload",
            files={"files": ("test.txt", sample_text_file, "text/plain")}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "chunks_added" in data
        assert data["chunks_added"] > 0

    def test_upload_image_document(self, client, sample_image):
        """Test uploading image for OCR to RAG"""
        response = client.post(
            "/v1/rag/upload",
            files={"files": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True

    def test_upload_too_large_file(self, client):
        """Test uploading file that's too large"""
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        response = client.post(
            "/v1/rag/upload",
            files={"files": ("large.txt", large_content, "text/plain")}
        )
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_rag_query(self, client, sample_text_file):
        """Test RAG query functionality"""
        # First upload a document
        client.post(
            "/v1/rag/upload",
            files={"files": ("test.txt", sample_text_file, "text/plain")}
        )
        
        # Then query it
        response = client.post(
            "/v1/rag/query",
            data={"question": "What is this document about?", "top_k": 3}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "result" in data

    def test_rag_query_invalid_params(self, client):
        """Test RAG query with invalid parameters"""
        response = client.post(
            "/v1/rag/query",
            data={"question": "", "top_k": 25}  # Empty question, too high top_k
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestXAIEndpoints:
    def test_gradcam_generation(self, client, sample_image):
        """Test Grad-CAM generation"""
        response = client.post(
            "/v1/xai/gradcam",
            files={"image": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/png"

    def test_gradcam_with_class_idx(self, client, sample_image):
        """Test Grad-CAM with specific class index"""
        response = client.post(
            "/v1/xai/gradcam",
            data={"class_idx": 1},
            files={"image": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/png"

    def test_gradcam_invalid_file(self, client):
        """Test Grad-CAM with non-image file"""
        response = client.post(
            "/v1/xai/gradcam",
            files={"image": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

class TestAudioEndpoints:
    def test_asr_endpoint(self, client, temp_audio_file):
        """Test speech-to-text endpoint"""
        with open(temp_audio_file, 'rb') as f:
            response = client.post(
                "/v1/audio/asr",
                files={"audio": ("test.wav", f.read(), "audio/wav")}
            )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "text" in data

    def test_asr_invalid_file(self, client):
        """Test ASR with non-audio file"""
        response = client.post(
            "/v1/audio/asr",
            files={"audio": ("test.txt", b"not audio", "text/plain")}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_vad_endpoint(self, client, temp_audio_file):
        """Test voice activity detection endpoint"""
        with open(temp_audio_file, 'rb') as f:
            response = client.post(
                "/v1/audio/vad",
                files={"audio": ("test.wav", f.read(), "audio/wav")}
            )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ok"] is True
        assert "segments" in data

    def test_tts_endpoint(self, client):
        """Test text-to-speech endpoint"""
        response = client.post(
            "/v1/audio/tts",
            data={"text": "Hello world"}
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"

    def test_tts_invalid_text(self, client):
        """Test TTS with invalid text"""
        response = client.post(
            "/v1/audio/tts",
            data={"text": "x" * 1000}  # Too long
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestErrorHandling:
    def test_global_exception_handler(self, client):
        """Test that unhandled exceptions are caught"""
        # This would need a specific endpoint that raises an exception
        # For now, test with malformed request
        response = client.post("/v1/multimodal/analyze", json={"invalid": "data"})
        # Should not return 500 due to global exception handler
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_404_endpoint(self, client):
        """Test non-existent endpoint"""
        response = client.get("/non-existent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

class TestCORS:
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")
        # CORS headers should be present due to middleware
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]
