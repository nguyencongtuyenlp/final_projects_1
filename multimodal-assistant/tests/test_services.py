"""
Tests for service modules
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import torch

class TestRAGStore:
    @pytest.fixture
    def temp_rag_dir(self):
        """Create temporary directory for RAG storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_rag_store_initialization(self, temp_rag_dir):
        """Test RAG store initialization"""
        from app.services.rag_store import RAGStore
        
        with patch('app.models.registry.get_sbert') as mock_sbert:
            mock_model = MagicMock()
            mock_sbert.return_value = mock_model
            
            store = RAGStore(base_dir=temp_rag_dir)
            assert store.base_dir == Path(temp_rag_dir)
            assert store.index is None  # No index initially

    def test_add_documents(self, temp_rag_dir):
        """Test adding documents to RAG store"""
        from app.services.rag_store import RAGStore
        
        with patch('app.models.registry.get_sbert') as mock_sbert:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
            mock_sbert.return_value = mock_model
            
            store = RAGStore(base_dir=temp_rag_dir)
            
            docs = [
                {"text": "Document 1", "meta": {"filename": "doc1.txt"}},
                {"text": "Document 2", "meta": {"filename": "doc2.txt"}}
            ]
            
            count = store.add_documents(docs)
            assert count == 2
            assert store.index is not None
            assert store.index.ntotal == 2

    def test_search_documents(self, temp_rag_dir):
        """Test searching documents"""
        from app.services.rag_store import RAGStore
        
        with patch('app.models.registry.get_sbert') as mock_sbert:
            mock_model = MagicMock()
            # Mock embeddings for documents and query
            mock_model.encode.side_effect = [
                np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),  # docs
                np.array([[0.2, 0.3, 0.4]], dtype=np.float32)  # query
            ]
            mock_sbert.return_value = mock_model
            
            store = RAGStore(base_dir=temp_rag_dir)
            
            # Add documents first
            docs = [
                {"text": "AI is amazing", "meta": {"type": "tech"}},
                {"text": "Cats are cute", "meta": {"type": "animals"}}
            ]
            store.add_documents(docs)
            
            # Search
            results = store.search("artificial intelligence", top_k=1)
            assert len(results) <= 1
            if results:
                assert "text" in results[0]
                assert "score" in results[0]

    def test_chunk_text(self):
        """Test text chunking functionality"""
        from app.services.rag_store import chunk_text
        
        # Short text - should return as is
        short_text = "Short text"
        chunks = chunk_text(short_text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Long text - should be chunked
        long_text = "x" * 1000
        chunks = chunk_text(long_text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # Check overlap
        assert chunks[1].startswith(chunks[0][-50:])

class TestRAGService:
    def test_rag_service_initialization(self):
        """Test RAG service initialization"""
        from app.services.rag_service import RAGService
        
        with patch('app.services.rag_store.RAGStore'), \
             patch('app.pipelines.nlp.NLPPipeline'):
            service = RAGService()
            assert service is not None

    def test_add_texts(self):
        """Test adding texts to RAG service"""
        from app.services.rag_service import RAGService
        
        with patch('app.services.rag_store.RAGStore') as mock_store_class, \
             patch('app.pipelines.nlp.NLPPipeline'):
            
            mock_store = MagicMock()
            mock_store.add_documents.return_value = 3
            mock_store_class.return_value = mock_store
            
            service = RAGService()
            
            texts = ["Document 1", "Document 2"]
            meta = {"source": "test"}
            
            count = service.add_texts(texts, meta)
            assert count == 3  # Assuming chunking created 3 chunks
            mock_store.add_documents.assert_called_once()

    def test_query_rag(self):
        """Test querying RAG service"""
        from app.services.rag_service import RAGService
        
        with patch('app.services.rag_store.RAGStore') as mock_store_class, \
             patch('app.pipelines.nlp.NLPPipeline') as mock_nlp_class:
            
            # Mock store
            mock_store = MagicMock()
            mock_store.search.return_value = [
                {"text": "Answer context", "score": 0.9}
            ]
            mock_store_class.return_value = mock_store
            
            # Mock NLP
            mock_nlp = MagicMock()
            mock_nlp.qa.return_value = {"answer": "The answer is 42"}
            mock_nlp_class.return_value = mock_nlp
            
            service = RAGService()
            result = service.query("What is the answer?", top_k=5)
            
            assert "answer" in result
            assert "hits" in result
            mock_store.search.assert_called_once_with("What is the answer?", top_k=5)

class TestOrchestrator:
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        from app.services.orchestrator import Orchestrator
        
        with patch('app.pipelines.multimodal.MultimodalPipeline'):
            orchestrator = Orchestrator()
            assert orchestrator is not None

    def test_analyze_delegation(self):
        """Test that orchestrator delegates to multimodal pipeline"""
        from app.services.orchestrator import Orchestrator
        from PIL import Image
        
        with patch('app.pipelines.multimodal.MultimodalPipeline') as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.run.return_value = {"result": "test"}
            mock_mm_class.return_value = mock_mm
            
            orchestrator = Orchestrator()
            
            img = Image.new('RGB', (100, 100))
            result = orchestrator.analyze(text="test", image=img, tasks=["ocr"])
            
            assert result == {"result": "test"}
            mock_mm.run.assert_called_once_with(text="test", image=img, tasks=["ocr"])

class TestXAI:
    def test_gradcam_initialization(self):
        """Test GradCAM initialization"""
        from app.xai.gradcam import GradCAM
        
        with patch('torchvision.models.resnet18') as mock_resnet:
            mock_model = MagicMock()
            mock_resnet.return_value = mock_model
            
            gradcam = GradCAM()
            assert gradcam is not None
            assert gradcam.target_layer_name == "layer4"

    def test_gradcam_generate(self):
        """Test GradCAM generation"""
        from app.xai.gradcam import GradCAM
        from PIL import Image
        
        with patch('torchvision.models.resnet18') as mock_resnet:
            # Mock model
            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model.return_value = torch.tensor([[0.1, 0.9, 0.2]])  # Logits
            
            # Mock layer for hooks
            mock_layer = MagicMock()
            mock_model.layer4 = mock_layer
            
            mock_resnet.return_value = mock_model
            
            gradcam = GradCAM()
            
            # Mock activations and gradients
            gradcam.activations = torch.randn(1, 512, 7, 7)
            gradcam.gradients = torch.randn(1, 512, 7, 7)
            
            img = Image.new('RGB', (224, 224))
            
            with patch('cv2.resize') as mock_resize:
                mock_resize.return_value = np.random.rand(224, 224)
                
                heatmap, class_idx = gradcam.generate(img)
                
                assert isinstance(heatmap, np.ndarray)
                assert isinstance(class_idx, int)

    def test_overlay_heatmap(self):
        """Test heatmap overlay function"""
        from app.xai.gradcam import overlay_heatmap
        from PIL import Image
        
        img = Image.new('RGB', (100, 100))
        heatmap = np.random.rand(224, 224)
        
        with patch('cv2.applyColorMap') as mock_colormap, \
             patch('cv2.cvtColor') as mock_cvt:
            
            mock_colormap.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_cvt.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            result = overlay_heatmap(img, heatmap)
            
            assert isinstance(result, Image.Image)
            assert result.size == (224, 224)

class TestUtils:
    def test_audio_utils_import(self):
        """Test audio utilities can be imported"""
        from app.utils.audio import load_audio_from_bytes, wav_bytes_from_tensor
        assert load_audio_from_bytes is not None
        assert wav_bytes_from_tensor is not None

    def test_auth_utils(self):
        """Test authentication utilities"""
        from app.utils.auth import require_bearer
        from fastapi import Request, HTTPException
        
        # Mock request without auth
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        
        with patch.dict('os.environ', {}, clear=True):
            # No auth token set - should pass
            result = require_bearer(mock_request)
            assert result is None
        
        with patch.dict('os.environ', {'APP_AUTH_TOKEN': 'test_token'}):
            # Auth required but missing header
            with pytest.raises(HTTPException):
                require_bearer(mock_request)
            
            # Valid auth header
            mock_request.headers = {"Authorization": "Bearer test_token"}
            result = require_bearer(mock_request)
            assert result is None
            
            # Invalid token
            mock_request.headers = {"Authorization": "Bearer wrong_token"}
            with pytest.raises(HTTPException):
                require_bearer(mock_request)
