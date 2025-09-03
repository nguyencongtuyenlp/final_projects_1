"""
Test configuration and fixtures
"""
import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import io

@pytest.fixture
def client():
    """FastAPI test client"""
    from app.server import app
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf.getvalue()

@pytest.fixture
def sample_text_file():
    """Create a sample text file"""
    text = "This is a sample document for testing RAG functionality. It contains information about AI and machine learning."
    return text.encode('utf-8')

@pytest.fixture
def sample_pdf_content():
    """Create mock PDF content (as bytes)"""
    # For testing, we'll just return some bytes that represent a PDF
    return b"%PDF-1.4 fake pdf content for testing"

@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing"""
    import wave
    import struct
    
    # Create a simple sine wave audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        with wave.open(f.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            
            # Generate 1 second of 440Hz sine wave
            duration = 1.0
            sample_rate = 16000
            samples = int(duration * sample_rate)
            
            for i in range(samples):
                value = int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate))
                wav_file.writeframes(struct.pack('<h', value))
                
        yield f.name
        
    # Cleanup
    os.unlink(f.name)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
