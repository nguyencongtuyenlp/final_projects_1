from typing import Dict, Any, Optional
import torch
from PIL import Image, ImageOps, ImageFilter
from app.models.registry import get_trocr, get_blip_vqa

class VisionPipeline:
    def __init__(self):
        self.trocr_processor, self.trocr_model = get_trocr()
        self.blip_processor, self.blip_model = get_blip_vqa()

    def _preprocess_variants(self, image: Image.Image):
        variants = []
        img = image.convert("L")  # grayscale
        variants.append(img)
        # Contrast & sharpen
        variants.append(ImageOps.autocontrast(img))
        variants.append(ImageOps.autocontrast(img.filter(ImageFilter.SHARPEN)))
        # Invert (for white text on dark)
        variants.append(ImageOps.invert(img))
        # Slight resize up to help small text
        variants.append(ImageOps.autocontrast(img.resize((int(img.width*1.25), int(img.height*1.25)))))
        return variants

    def ocr(self, image: Image.Image) -> str:
        # Try multiple preprocess variants, return the longest non-empty text
        candidates = []
        for variant in self._preprocess_variants(image):
            pixel_values = self.trocr_processor(images=variant.convert("RGB"), return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if text:
                candidates.append(text)
        if not candidates:
            return ""
        # Heuristic: choose the one with max length (more content)
        return max(candidates, key=len)

    def vqa(self, image: Image.Image, question: str) -> Dict[str, Any]:
        inputs = self.blip_processor(images=image, text=question, return_tensors="pt")
        out_ids = self.blip_model.generate(**inputs, max_new_tokens=20)
        answer = self.blip_processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return {"answer": answer}

    def caption(self, image: Image.Image) -> str:
        # Use BLIP VQA with a generic prompt to produce a caption-like answer
        prompt = "Describe the image"
        return self.vqa(image, prompt)["answer"]
