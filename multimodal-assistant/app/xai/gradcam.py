from __future__ import annotations
from typing import Tuple
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

class GradCAM:
    def __init__(self, target_layer: str = "layer4"):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.target_layer_name = target_layer
        self.activations = None
        self.gradients = None
        self._register()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def _get_target_layer(self):
        return getattr(self.model, self.target_layer_name)

    def _register(self):
        layer = self._get_target_layer()
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def _preprocess(self, img: Image.Image):
        return self.tf(img).unsqueeze(0)

    def generate(self, img: Image.Image, class_idx: int | None = None) -> Tuple[np.ndarray, int]:
        x = self._preprocess(img)
        x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        grads = self.gradients  # [B,C,H,W]
        acts = self.activations
        weights = torch.mean(grads, dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = torch.sum(weights * acts, dim=1)  # [B,H,W]
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        cam = cv2.resize(cam, (224, 224))
        return cam, class_idx

def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    img = img.resize((224,224)).convert("RGB")
    heatmap_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_color + (1 - alpha) * np.array(img)).astype(np.uint8)
    return Image.fromarray(blended)
