from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from .retrieval import SemanticRetriever
from app.models.registry import get_summarizer, get_qa_reader

class NLPPipeline:
    def __init__(self):
        self.retriever = SemanticRetriever()

    def summarize(self, text: str, max_len: int = 140) -> str:
        summarizer = get_summarizer()
        out = summarizer(text, max_length=max_len, min_length=50, do_sample=False)
        return out[0]["summary_text"]

    def qa(self, question: str, context: str) -> Dict[str, Any]:
        tok, mdl = get_qa_reader()
        inputs = tok(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = mdl(**inputs)
        start_idx = int(torch.argmax(outputs.start_logits))
        end_idx = int(torch.argmax(outputs.end_logits))
        answer = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1]))
        return {"answer": answer, "start": start_idx, "end": end_idx}

    def rag(self, question: str, corpus: List[str], top_k: int = 3) -> Dict[str, Any]:
        docs = self.retriever.search(question, corpus, top_k=top_k)
        best_ctx = docs[0]["text"] if docs else ""
        ans = self.qa(question, best_ctx)
        ans["context"] = best_ctx
        ans["retrieved"] = docs
        return ans
