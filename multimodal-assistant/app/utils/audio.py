from __future__ import annotations
from typing import Tuple
import io, wave, struct
import torch
import torchaudio

def load_audio_from_bytes(b: bytes, target_sr: int = 16000) -> torch.Tensor:
    '''Return mono waveform tensor [1, T] at target_sr.'''
    bio = io.BytesIO(b)
    wav, sr = torchaudio.load(bio)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    # clamp to [-1, 1]
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav

def wav_bytes_from_tensor(waveform: torch.Tensor, sr: int = 16000) -> bytes:
    '''Encode float tensor [-1,1] to PCM16 WAV bytes.'''
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    # scale to int16
    x = (waveform * 32767.0).to(torch.int16).cpu().numpy().tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x)
    return bio.getvalue()
