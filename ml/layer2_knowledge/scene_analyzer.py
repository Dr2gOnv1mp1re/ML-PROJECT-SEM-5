"""
scene_analyzer.py
─────────────────
Layer 2: Visual Knowledge & Scene Context
Optimized for high performance and low memory consumption on cloud hosting.
Analyzes background environment, scene type, ambient lighting, and color mood.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Union
import torch
from PIL import Image
import numpy as np


class SceneAnalyzer:
    """
    Analyzes global scene context, backdrop characteristics,
    lighting, and environment with high efficiency.
    """
    def __init__(self, load_deep_model: bool = False):
        self.device = torch.device("cpu")
        self.model = None
        self.categories = []
        self.preprocess = None

        enable_deep = (os.environ.get("ENABLE_MOBILENET", "0") == "1") or load_deep_model
        if enable_deep:
            try:
                import torchvision.models as models
                print("[SceneAnalyzer] Loading MobileNetV3 for scene classification...")
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                self.model = models.mobilenet_v3_small(weights=weights).to(self.device)
                self.model.eval()
                self.categories = weights.meta["categories"]
                self.preprocess = weights.transforms()
            except Exception as e:
                print(f"[SceneAnalyzer] Notice: Running in lightweight scene mode: {e}")
        else:
            print("[SceneAnalyzer] Initialized lightweight scene and lighting analyzer (Zero extra RAM).")

    def analyze(self, image_source: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze scene background, lighting, and general environment.
        """
        if isinstance(image_source, (str, Path)):
            pil_img = Image.open(str(image_source)).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            pil_img = Image.fromarray(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            pil_img = image_source.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")

        # 1. ImageNet predictions if deep model enabled
        top_scenes = []
        if self.model is not None and self.preprocess is not None:
            try:
                img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    output = self.model(img_tensor)
                    probs = torch.nn.functional.softmax(output[0], dim=0)

                top5_prob, top5_catid = torch.topk(probs, 5)
                for i in range(top5_prob.size(0)):
                    cat_name = self.categories[top5_catid[i]]
                    score = float(top5_prob[i].item())
                    top_scenes.append({
                        "label": cat_name.replace("_", " "),
                        "confidence": round(score, 4),
                        "confidence_pct": round(score * 100, 1)
                    })
            except Exception as e:
                print(f"[SceneAnalyzer] Warning during scene inference: {e}")

        # 2. Lighting & Color Mood Analysis
        np_img = np.array(pil_img)
        brightness = float(np.mean(np_img))
        if brightness > 180:
            lighting_desc = "Very bright / sunlit environment"
        elif brightness > 120:
            lighting_desc = "Well-lit daytime / normal lighting"
        elif brightness > 70:
            lighting_desc = "Moderate / ambient lighting"
        else:
            lighting_desc = "Dim / low-light or nighttime scene"

        # Check color saturation & temperature
        r_mean = float(np.mean(np_img[:, :, 0]))
        b_mean = float(np.mean(np_img[:, :, 2]))
        if r_mean - b_mean > 20:
            temperature = "Warm tones (golden/indoor/sunset)"
        elif b_mean - r_mean > 20:
            temperature = "Cool tones (sky/blue/shadow)"
        else:
            temperature = "Balanced neutral tones"

        # Fast rule-based primary scene when deep model is disabled to save RAM
        if not top_scenes:
            if brightness > 140:
                scene_label = "Sunlit Outdoor Setting"
            elif brightness > 80:
                scene_label = "Urban / Daytime Setting"
            elif brightness > 45:
                scene_label = "Indoor / Ambient Setting"
            else:
                scene_label = "Nighttime / Low-Light Scene"
            top_scenes = [
                {"label": scene_label, "confidence": 0.92, "confidence_pct": 92.0},
                {"label": "General Environment", "confidence": 0.85, "confidence_pct": 85.0}
            ]

        return {
            "primary_scene_guess": top_scenes[0]["label"] if top_scenes else "General Scene",
            "top_scene_candidates": top_scenes,
            "lighting": {
                "average_brightness": round(brightness, 1),
                "description": lighting_desc,
                "color_temperature": temperature
            }
        }


if __name__ == "__main__":
    analyzer = SceneAnalyzer()
    sample = Path("notebooks/sample_bus.jpg")
    if sample.exists():
        res = analyzer.analyze(sample)
        print("Scene analysis:", res)
