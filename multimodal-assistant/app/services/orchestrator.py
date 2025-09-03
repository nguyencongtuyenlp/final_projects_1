from typing import Dict, Any, List, Optional
from PIL import Image
from app.pipelines.multimodal import MultimodalPipeline

class Orchestrator:
    def __init__(self):
        self.mm = MultimodalPipeline()

    def analyze(self, text: Optional[str], image: Optional[Image.Image], tasks: List[str]) -> Dict[str, Any]:
        return self.mm.run(text=text, image=image, tasks=tasks or [])
