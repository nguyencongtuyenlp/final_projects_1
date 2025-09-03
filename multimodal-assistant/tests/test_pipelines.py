"""
Tests for pipeline modules
"""
import pytest
from PIL import Image
import numpy as np
import torch
from unittest.mock import patch, MagicMock

class TestNLPPipeline:
    def test_nlp_pipeline_import(self):
        """Test NLP pipeline can be imported"""
        from app.pipelines.nlp import NLPPipeline
        pipeline = NLPPipeline()
        assert pipeline is not None

    @patch('app.models.registry.get_summarizer')
    def test_summarize(self, mock_summarizer):
        """Test text summarization"""
        from app.pipelines.nlp import NLPPipeline
        
        # Mock the summarizer
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "This is a summary"}]
        mock_summarizer.return_value = mock_pipe
        
        pipeline = NLPPipeline()
        result = pipeline.summarize("This is a long text that needs summarization.")
        
        assert result == "This is a summary"
        mock_pipe.assert_called_once()

    @patch('app.models.registry.get_qa_reader')
    def test_qa(self, mock_qa_reader):
        """Test question answering"""
        from app.pipelines.nlp import NLPPipeline
        
        # Mock the QA model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_qa_reader.return_value = (mock_tokenizer, mock_model)
        
        # Mock tokenizer and model outputs
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        mock_tokenizer.convert_tokens_to_string.return_value = "answer"
        mock_tokenizer.convert_ids_to_tokens.return_value = ["answer"]
        
        mock_outputs = MagicMock()
        mock_outputs.start_logits = torch.tensor([[0.1, 0.9, 0.2]])
        mock_outputs.end_logits = torch.tensor([[0.2, 0.1, 0.8]])
        mock_model.return_value = mock_outputs
        
        pipeline = NLPPipeline()
        result = pipeline.qa("What is the answer?", "The answer is 42.")
        
        assert "answer" in result
        assert "start" in result
        assert "end" in result

class TestVisionPipeline:
    def test_vision_pipeline_import(self):
        """Test vision pipeline can be imported"""
        from app.pipelines.vision import VisionPipeline
        pipeline = VisionPipeline()
        assert pipeline is not None

    @patch('app.models.registry.get_trocr')
    def test_ocr(self, mock_trocr):
        """Test OCR functionality"""
        from app.pipelines.vision import VisionPipeline
        
        # Mock TrOCR
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_trocr.return_value = (mock_processor, mock_model)
        
        mock_processor.return_value.pixel_values = torch.randn(1, 3, 224, 224)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.return_value = ["extracted text"]
        
        pipeline = VisionPipeline()
        img = Image.new('RGB', (100, 100))
        result = pipeline.ocr(img)
        
        assert result == "extracted text"

    @patch('app.models.registry.get_blip_vqa')
    def test_vqa(self, mock_blip):
        """Test visual question answering"""
        from app.pipelines.vision import VisionPipeline
        
        # Mock BLIP VQA
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_blip.return_value = (mock_processor, mock_model)
        
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_processor.batch_decode.return_value = ["answer text"]
        
        pipeline = VisionPipeline()
        img = Image.new('RGB', (100, 100))
        result = pipeline.vqa(img, "What is in the image?")
        
        assert "answer" in result
        assert result["answer"] == "answer text"

class TestAudioPipeline:
    def test_audio_pipeline_import(self):
        """Test audio pipeline can be imported"""
        from app.pipelines.audio import AudioPipeline
        pipeline = AudioPipeline()
        assert pipeline is not None

    @patch('app.models.registry.get_asr')
    def test_asr(self, mock_asr):
        """Test automatic speech recognition"""
        from app.pipelines.audio import AudioPipeline
        
        # Mock ASR pipeline
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": "transcribed text"}
        mock_asr.return_value = mock_pipe
        
        pipeline = AudioPipeline()
        
        # Mock audio loading
        with patch('app.utils.audio.load_audio_from_bytes') as mock_load:
            mock_load.return_value = torch.randn(1, 16000)  # 1 second of audio
            result = pipeline.asr(b"fake audio bytes")
        
        assert "text" in result
        assert result["text"] == "transcribed text"

    def test_vad_without_webrtc(self):
        """Test VAD without webrtcvad (fallback)"""
        from app.pipelines.audio import AudioPipeline
        
        pipeline = AudioPipeline()
        pipeline.vad = None  # Simulate webrtcvad not available
        
        with patch('app.utils.audio.load_audio_from_bytes') as mock_load:
            # High energy audio should be detected
            mock_load.return_value = torch.randn(1, 16000) * 0.1  # High amplitude
            result = pipeline.vad_segments(b"fake audio", sample_rate=16000)
            
        assert "segments" in result
        # Should detect at least one segment due to energy

    @patch('pyttsx3.init')
    def test_tts_success(self, mock_pyttsx3):
        """Test TTS with successful synthesis"""
        from app.pipelines.audio import AudioPipeline
        
        # Mock pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        pipeline = AudioPipeline()
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('builtins.open', create=True) as mock_open, \
             patch('os.remove') as mock_remove:
            
            mock_temp.return_value.__enter__.return_value.name = "test.wav"
            mock_open.return_value.read.return_value = b"fake audio data"
            
            result = pipeline.tts("Hello world")
            
        assert isinstance(result, bytes)

    def test_tts_fallback(self):
        """Test TTS fallback when pyttsx3 fails"""
        from app.pipelines.audio import AudioPipeline
        
        pipeline = AudioPipeline()
        
        # Should return silence when TTS fails
        result = pipeline.tts("Hello world")
        assert isinstance(result, bytes)
        assert len(result) > 0

class TestMultimodalPipeline:
    def test_multimodal_pipeline_import(self):
        """Test multimodal pipeline can be imported"""
        from app.pipelines.multimodal import MultimodalPipeline
        pipeline = MultimodalPipeline()
        assert pipeline is not None

    @patch('app.pipelines.vision.VisionPipeline.ocr')
    def test_ocr_task(self, mock_ocr):
        """Test OCR task in multimodal pipeline"""
        from app.pipelines.multimodal import MultimodalPipeline
        
        mock_ocr.return_value = "extracted text"
        
        pipeline = MultimodalPipeline()
        img = Image.new('RGB', (100, 100))
        result = pipeline.run(text=None, image=img, tasks=["ocr"])
        
        assert "ocr" in result
        assert result["ocr"] == "extracted text"

    @patch('app.pipelines.nlp.NLPPipeline.summarize')
    def test_summary_task(self, mock_summarize):
        """Test summarization task"""
        from app.pipelines.multimodal import MultimodalPipeline
        
        mock_summarize.return_value = "summary text"
        
        pipeline = MultimodalPipeline()
        result = pipeline.run(text="Long text to summarize", image=None, tasks=["summary"])
        
        assert "summary" in result
        assert result["summary"] == "summary text"

    def test_qa_without_context(self):
        """Test QA task without context (should return error)"""
        from app.pipelines.multimodal import MultimodalPipeline
        
        pipeline = MultimodalPipeline()
        result = pipeline.run(text="What is the answer?", image=None, tasks=["qa"])
        
        assert "qa" in result
        assert "error" in result["qa"]

class TestStreamingPipeline:
    def test_streaming_asr_import(self):
        """Test streaming ASR can be imported"""
        from app.pipelines.streaming import StreamingASR
        stream = StreamingASR()
        assert stream is not None

    def test_streaming_reset(self):
        """Test streaming ASR reset"""
        from app.pipelines.streaming import StreamingASR
        
        stream = StreamingASR()
        stream.buffer = torch.randn(1000)  # Add some data
        stream.reset()
        
        assert len(stream.buffer) == 0
        assert stream.t0_cursor == 0.0

class TestRetrieval:
    def test_semantic_retriever_import(self):
        """Test semantic retriever can be imported"""
        from app.pipelines.retrieval import SemanticRetriever
        retriever = SemanticRetriever()
        assert retriever is not None

    @patch('app.models.registry.get_sbert')
    def test_semantic_search(self, mock_sbert):
        """Test semantic search functionality"""
        from app.pipelines.retrieval import SemanticRetriever
        
        # Mock sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_sbert.return_value = mock_model
        
        retriever = SemanticRetriever()
        corpus = ["doc1", "doc2", "doc3"]
        results = retriever.search("query", corpus, top_k=2)
        
        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)