from typing import List, Dict, Any
from app.services.rag_store import RAGStore, chunk_text
from app.pipelines.nlp import NLPPipeline

class RAGService:
    def __init__(self):
        self.store = RAGStore()
        self.nlp = NLPPipeline()

    def add_texts(self, texts: List[str], meta: Dict[str, Any]) -> int:
        docs = []
        for t in texts:
            for ch in chunk_text(t):
                docs.append({"text": ch, "meta": meta})
        return self.store.add_documents(docs)

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        hits = self.store.search(question, top_k=top_k)
        ctx = "\n\n".join([h["text"] for h in hits])
        answer = self.nlp.qa(question, ctx) if ctx else {"answer": "", "error": "No context"}
        return {"answer": answer, "hits": hits}
