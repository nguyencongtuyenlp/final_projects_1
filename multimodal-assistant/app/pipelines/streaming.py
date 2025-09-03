from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from app.pipelines.audio import AudioPipeline
from app.utils.audio import load_audio_from_bytes

class StreamingASR:
    """ASR segmentation by VAD; accumulate audio and emit segments."""
    def __init__(self, sample_rate: int = 16000, min_speech_sec: float = 0.6, max_buffer_sec: float = 20.0):
        self.sample_rate = sample_rate
        self.min_speech_sec = min_speech_sec
        self.max_buffer_sec = max_buffer_sec
        self.pipeline = AudioPipeline()
        self.buffer = torch.zeros(0)  # mono
        self.t0_cursor = 0.0  # seconds from start
        self.in_speech = False

    def reset(self):
        self.buffer = torch.zeros(0)
        self.t0_cursor = 0.0
        self.in_speech = False

    def push_bytes(self, b: bytes) -> List[Dict[str, Any]]:
        """Return list of transcripts for completed segments."""
        # append audio
        wav = load_audio_from_bytes(b, target_sr=self.sample_rate).squeeze(0)  # [T]
        self.buffer = torch.cat([self.buffer, wav], dim=0)
        # run VAD over current buffer
        results = []
        segs = self.pipeline.vad_segments(self._to_wav_bytes(self.buffer), sample_rate=self.sample_rate)["segments"]
        # Emit any full ending segments except the last open one
        for i, (s, e) in enumerate(segs[:-1] if segs else []):
            # segment indices
            tvec = torch.arange(len(self.buffer)) / self.sample_rate
            s_idx = int(s * self.sample_rate)
            e_idx = int(e * self.sample_rate)
            if e - s >= self.min_speech_sec:
                seg = self.buffer[s_idx:e_idx].clone()
                transcript = self.pipeline.asr(self._to_wav_bytes(seg))["text"]
                results.append({"type": "final", "text": transcript, "t0": round(self.t0_cursor + s, 3), "t1": round(self.t0_cursor + e, 3)})
        # Trim emitted segments from buffer to avoid reprocessing
        if segs:
            last_end = segs[-1][1] if len(segs[-1]) == 2 else segs[-1][0]
            cut = int(last_end * self.sample_rate)
            if cut > 0 and cut < len(self.buffer):
                # keep tail for continuity
                self.buffer = self.buffer[cut:]
                self.t0_cursor += last_end
        # Hard flush if buffer too long
        if len(self.buffer) > int(self.max_buffer_sec * self.sample_rate):
            transcript = self.pipeline.asr(self._to_wav_bytes(self.buffer))["text"]
            results.append({"type": "final", "text": transcript, "t0": round(self.t0_cursor, 3), "t1": round(self.t0_cursor + len(self.buffer)/self.sample_rate, 3)})
            self.reset()
        return results

    def flush(self) -> Optional[Dict[str, Any]]:
        if len(self.buffer) == 0:
            return None
        transcript = self.pipeline.asr(self._to_wav_bytes(self.buffer))["text"]
        out = {"type": "final", "text": transcript, "t0": round(self.t0_cursor, 3), "t1": round(self.t0_cursor + len(self.buffer)/self.sample_rate, 3)}
        self.reset()
        return out

    def _to_wav_bytes(self, wav: torch.Tensor) -> bytes:
        from app.utils.audio import wav_bytes_from_tensor
        return wav_bytes_from_tensor(wav.unsqueeze(0), sr=self.sample_rate)
