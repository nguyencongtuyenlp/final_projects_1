from typing import Dict, Any, List, Optional
from PIL import Image
from .nlp import NLPPipeline
from .vision import VisionPipeline

class MultimodalPipeline:
    def __init__(self):
        self.nlp = NLPPipeline()
        self.vision = VisionPipeline()

    def run(self, *, text: Optional[str], image: Optional[Image.Image], tasks: List[str]) -> Dict[str, Any]:
        res = {}

        if "ocr" in tasks and image is not None:
            res["ocr"] = self.vision.ocr(image)

        if "vqa" in tasks and image is not None and text:
            res["vqa"] = self.vision.vqa(image, text)

        # Image captioning
        if "caption" in tasks and image is not None:
            try:
                res["caption"] = self.vision.caption(image)
            except Exception:
                res["caption"] = self.vision.vqa(image, "Describe the image")["answer"]

            # Nếu caption quá chung chung (ví dụ 'text'), tự động kèm OCR
            simple_caps = {"text", "document", "paper", "screenshot"}
            if not res.get("ocr") and (
                res.get("caption", "").strip().lower() in simple_caps or len(res.get("caption", "")) < 5
            ):
                res["ocr"] = self.vision.ocr(image)

        if "summary" in tasks and text:
            res["summary"] = self.nlp.summarize(text)

        if "qa" in tasks and text:
            # Nếu có OCR thì dùng làm context
            ctx = res.get("ocr", "")
            if ctx:
                res["qa"] = self.nlp.qa(text, ctx)
            else:
                res["qa"] = {"error": "Thiếu context cho QA. Hãy gửi ảnh tài liệu (OCR) hoặc cung cấp context."}

        return res
