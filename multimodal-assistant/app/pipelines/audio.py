from __future__ import annotations
from typing import List, Dict, Any, Tuple
import io, struct
import numpy as np
import torch

from app.models.registry import get_asr
from app.utils.audio import load_audio_from_bytes, wav_bytes_from_tensor

class AudioPipeline:
    def __init__(self):
        self.asr_pipe = get_asr()
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # 0-3
        except Exception:
            self.vad = None

    # ------------- ASR -------------
    def asr(self, audio_bytes: bytes) -> Dict[str, Any]:
        wav = load_audio_from_bytes(audio_bytes, target_sr=16000)  # [1,T]
        # to numpy float32 for HF pipeline
        arr = wav.squeeze(0).cpu().numpy()
        text = self.asr_pipe(arr, chunk_length_s=20, stride_length_s=2)["text"]
        return {"text": text}

    # ------------- VAD -------------
    def _frame_generator(self, pcm16: bytes, sample_rate: int, frame_ms: int = 30):
        n = int(sample_rate * (frame_ms / 1000.0) * 2)  # bytes per frame (mono, 16-bit)
        for i in range(0, len(pcm16), n):
            yield pcm16[i:i+n]

    def vad_segments(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        import wave, io
        wav = load_audio_from_bytes(audio_bytes, target_sr=sample_rate)
        pcm = (wav.squeeze(0).mul(32767).to(torch.int16).cpu().numpy().tobytes())
        if self.vad is None:
            # naive energy VAD: return single segment if energy > threshold
            energy = float(torch.mean(wav**2))
            if energy > 1e-4:
                return {"segments": [[0.0, len(wav.squeeze(0))/sample_rate]]}
            return {"segments": []}
        frame_ms = 30
        frames = list(self._frame_generator(pcm, sample_rate, frame_ms))
        segs = []
        in_speech = False
        start = 0.0
        for i, fr in enumerate(frames):
            is_speech = self.vad.is_speech(fr, sample_rate) if len(fr) == int(sample_rate*(frame_ms/1000.0)*2) else False
            t0 = i * frame_ms / 1000.0
            t1 = (i+1) * frame_ms / 1000.0
            if is_speech and not in_speech:
                in_speech = True
                start = t0
            elif not is_speech and in_speech:
                in_speech = False
                segs.append([round(start,3), round(t0,3)])
        if in_speech:
            segs.append([round(start,3), round(len(frames)*frame_ms/1000.0,3)])
        return {"segments": segs}

    # ------------- TTS -------------
    def tts(self, text: str) -> bytes:
        try:
            import pyttsx3, tempfile, os
            engine = pyttsx3.init()
            # Optionally tweak voice/rate here
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp = tf.name
            engine.save_to_file(text, tmp)
            engine.runAndWait()
            data = open(tmp, "rb").read()
            os.remove(tmp)
            return data
        except Exception:
            # Fallback: generate silence placeholder 0.5s
            sr = 16000
            t = torch.zeros(int(sr*0.5))
            return wav_bytes_from_tensor(t, sr=sr)
