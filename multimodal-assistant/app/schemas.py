"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from fastapi import UploadFile

class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    tasks: List[str] = []
    
    @validator('tasks')
    def validate_tasks(cls, v):
        valid_tasks = {'ocr', 'vqa', 'summary', 'qa', 'caption'}
        for task in v:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task: {task}. Valid tasks: {valid_tasks}")
        return v

class AnalyzeResponse(BaseModel):
    ok: bool = True
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class RAGQueryResponse(BaseModel):
    ok: bool = True
    result: Dict[str, Any]
    error: Optional[str] = None

class ASRResponse(BaseModel):
    ok: bool = True
    text: str
    error: Optional[str] = None

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)

class VADResponse(BaseModel):
    ok: bool = True
    segments: List[List[float]]
    error: Optional[str] = None

class GradCAMRequest(BaseModel):
    class_idx: Optional[int] = None

class APIError(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None
