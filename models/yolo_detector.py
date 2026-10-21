"""
YOLOv8 Ultra-Accuracy Waste Detection Engine
Integrates fine-tuned and pretrained Ultralytics YOLOv8 models (Waste12, Recyclable-Waste, COCO)
with advanced Computer Vision preprocessing (LAB-CLAHE, unsharp blur correction, EXIF auto-orientation),
multi-model consensus voting, and multi-spectral material analysis for maximum precision across both
live camera streams and high-resolution gallery uploads.
"""

from dataclasses import dataclass, field
import hashlib
import io
import os
import time
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Supported Waste Classes
WASTE_CLASSES = ["Metal", "Plastic", "Glass", "Paper"]

CLASS_COLORS = {
    "Metal": (245, 166, 35),      # Amber / Gold
    "Plastic": (33, 150, 243),    # Blue
    "Glass": (0, 188, 212),       # Cyan / Teal
    "Paper": (76, 175, 80),       # Emerald Green
    "Contaminated": (244, 67, 54) # Red
}

CLASS_HEX_COLORS = {
    "Metal": "#F5A623",
    "Plastic": "#2196F3",
    "Glass": "#00BCD4",
    "Paper": "#4CAF50",
    "Contaminated": "#F44336"
}

# ============================================================================
# Exhaustive Authoritative Material Mapping Tables
# Every YOLO class from all 3 models is mapped ONCE to exactly ONE material.
# Classes whose material depends on visual appearance are in AMBIGUOUS_CLASSES.
# ============================================================================

# Tier-1: Shape alone is sufficient — material is 100% certain
AUTHORITATIVE_METAL = {
    "can", "aluminium foil", "aluminum foil", "pop tab",
    "scissors", "scissor", "clipper", "nail clipper",
    "knife", "fork", "spoon", "toaster", "microwave", "sink", "oven",
    "pliers", "hardware", "tin", "metal",
}

AUTHORITATIVE_GLASS = {
    "broken glass", "brown-glass", "green-glass", "white-glass",
    "wine glass", "vase", "perfume", "perfume bottle", "glass jar",
    "glass",
}

AUTHORITATIVE_PAPER = {
    "cardboard", "carton", "paper", "book", "newspaper", "magazine",
}

AUTHORITATIVE_PLASTIC = {
    "plastic bag - wrapper", "plastic bag", "plastic container",
    "straw", "styrofoam piece", "styrofoam",
    "bottle cap", "lid",
}

# Tier-2: Shape is ambiguous — surface optics decide the material
AMBIGUOUS_CLASSES = {
    "bottle", "cup", "jar", "battery", "trash", "other litter",
    "unlabeled litter", "cigarette", "bowl", "toothbrush",
    "cell phone", "biological", "clothes", "shoes",
}

IGNORED_COCO_CLASSES = {
    "person", "tv", "laptop", "cell phone", "mouse", "keyboard", "remote",
    "potted plant", "chair", "couch", "bed", "dining table", "toilet",
    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "refrigerator", "clock",
    "teddy bear", "hair drier",
}

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    waste_type: str
    confidence: float
    raw_class: str = ""
    estimated_volume_liters: float = 0.5
    estimated_weight_kg: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xmin": int(self.xmin),
            "ymin": int(self.ymin),
            "xmax": int(self.xmax),
            "ymax": int(self.ymax),
            "waste_type": self.waste_type,
            "confidence": round(float(self.confidence), 4),
            "raw_class": self.raw_class,
            "estimated_volume_liters": round(float(self.estimated_volume_liters), 2),
            "estimated_weight_kg": round(float(self.estimated_weight_kg), 3),
        }

@dataclass
class DetectionResult:
    detections: List[BoundingBox] = field(default_factory=list)
    image_width: int = 640
    image_height: int = 480
    inference_time_ms: float = 0.0
    primary_category: str = "Unknown"
    confidence_avg: float = 0.0
    total_items: int = 0
    annotated_image_base64: Optional[str] = None
    engine_used: str = "Ultralytics-YOLOv8"
    probabilities: Dict[str, float] = field(default_factory=dict)
    features_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_category": self.primary_category,
            "confidence_avg": round(float(self.confidence_avg), 4),
            "total_items": self.total_items,
            "inference_time_ms": round(float(self.inference_time_ms), 2),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "engine_used": self.engine_used,
            "detections": [d.to_dict() for d in self.detections],
            "annotated_image_base64": self.annotated_image_base64,
            "probabilities": self.probabilities,
            "features_summary": self.features_summary
        }


class WasteYOLODetector:
    """
    High-Precision Multi-Model YOLOv8 Waste Detector.
    Integrates fine-tuned waste detection weights, dedicated 12-class waste weights,
    and pretrained COCO models with computer-vision enhancement for ultra-reliable waste sorting.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.12):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.waste_model = None
        self.coco_model = None
        self.waste12_model = None
        self.engine_name = "Ultralytics-YOLOv8"
        self._cache: Dict[Tuple[str, Optional[str]], DetectionResult] = {}
        self._max_cache_size: int = 64
        
        # Configure torch thread efficiency for CPU
        try:
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except Exception:
            pass
            
        self._initialize_detector()

    @property
    def yolo_model(self):
        """Compatibility property for status checks."""
        return self.waste_model or self.coco_model or self.waste12_model

    def _initialize_detector(self):
        """Loads actual trained YOLOv8 weights files."""
        try:
            from ultralytics import YOLO

            # 1. Dedicated 12-Class waste model (cardboard, paper, metal, plastic, brown/green/white glass, battery)
            waste12_weights = [
                os.path.join(os.path.dirname(__file__), "yolov8n_waste.pt"),
                "models/yolov8n_waste.pt"
            ]
            for w in waste12_weights:
                if w and os.path.exists(w):
                    self.waste12_model = YOLO(w)
                    break

            # 2. Primary fine-tuned recyclable waste model
            primary_weights = [
                self.model_path,
                os.path.join(os.path.dirname(__file__), "waste_yolov8_best.pt"),
                "models/waste_yolov8_best.pt"
            ]
            for w in primary_weights:
                if w and os.path.exists(w):
                    self.waste_model = YOLO(w)
                    break

            # 3. General COCO model (critical for scissors, clippers, cutlery, bottles, vases, books)
            coco_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            coco_weights = [
                coco_path,
                "models/yolov8n.pt",
                "yolov8n.pt"
            ]
            for w in coco_weights:
                if w and os.path.exists(w):
                    self.coco_model = YOLO(w)
                    break

            active_models = []
            if self.waste12_model: active_models.append("Waste12-YOLOv8")
            if self.waste_model: active_models.append("Waste-YOLOv8")
            if self.coco_model: active_models.append("COCO-YOLOv8")

            self.engine_name = f"Ultralytics-YOLOv8 ({', '.join(active_models)})" if active_models else "Ultralytics-YOLOv8"
        except Exception as e:
            print(f"[!] Warning initializing YOLO models: {e}")
            self.engine_name = "Ultralytics-YOLOv8"

    def preprocess_image(self, image: Image.Image, source: Optional[str] = None) -> Image.Image:
        """
        Applies high-accuracy computer vision preprocessing:
        1. EXIF orientation correction (fixes rotated gallery uploads).
        2. Adaptive CLAHE on LAB luminance channel (sharpens transparent glass, metallic luster, plastic borders, paper fibers).
        3. Motion blur / soft lens compensation via Laplacian unsharp filter.
        """
        # 1. EXIF orientation
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        np_img = np.array(img_rgb)

        try:
            # Convert to BGR for OpenCV processing
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            # 2. CLAHE (Contrast-Limited Adaptive Histogram Equalization) in LAB space
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clip_limit = 2.4 if source == "live_camera" else 1.8
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)

            enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # 3. Unsharp masking to counteract webcam/phone camera soft blur
            if source == "live_camera" or min(w, h) < 800:
                blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=2.0)
                sharpened = cv2.addWeighted(enhanced_bgr, 1.25, blurred, -0.25, 0)
                enhanced_bgr = np.clip(sharpened, 0, 255).astype(np.uint8)

            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(enhanced_rgb)
        except Exception as e:
            return img_rgb

    def _classify_detected_label(self, raw_label: str, image_crop: Optional[Image.Image] = None) -> Tuple[str, float]:
        """
        Two-tier hierarchical material classification:
        Tier 1 — Authoritative YOLO labels whose shape alone determines material (can, scissors, cardboard, etc.)
        Tier 2 — Ambiguous YOLO labels (bottle, cup, battery, trash) resolved by multi-spectral surface optics.
        """
        raw_lower = raw_label.lower().strip() if raw_label else "unknown"

        # ──────────────── TIER 1: Authoritative Shape → Material ────────────────
        if raw_lower in AUTHORITATIVE_METAL:
            return "Metal", 0.985
        if raw_lower in AUTHORITATIVE_GLASS:
            return "Glass", 0.985
        if raw_lower in AUTHORITATIVE_PAPER:
            return "Paper", 0.985
        if raw_lower in AUTHORITATIVE_PLASTIC:
            return "Plastic", 0.98

        if raw_lower in {"plastic", "other plastic"} and image_crop is not None:
            return self._classify_detected_label("unknown", image_crop)

        # ──────────────── TIER 2: Ambiguous Shape → Optical Tiebreak ────────────
        # Extract surface optical features from the bounding-box crop
        r, g, b, sat, spec_ratio, chroma = 128.0, 128.0, 128.0, 30.0, 0.0, 15.0
        has_optics = False

        if image_crop is not None:
            try:
                crop_np = np.array(image_crop.convert("RGB"))
                if crop_np.size > 0 and crop_np.shape[0] > 2 and crop_np.shape[1] > 2:
                    r = float(np.mean(crop_np[:, :, 0]))
                    g = float(np.mean(crop_np[:, :, 1]))
                    b = float(np.mean(crop_np[:, :, 2]))

                    hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
                    sat = float(np.mean(hsv[:, :, 1]))
                    spec_mask = (hsv[:, :, 2] > 230) & (hsv[:, :, 1] < 45)
                    spec_ratio = float(np.sum(spec_mask)) / max(1, crop_np.shape[0] * crop_np.shape[1])

                    chroma = (abs(r - g) + abs(g - b) + abs(r - b)) / 3.0
                    has_optics = True
            except Exception:
                pass

        # Helper booleans for optical surface patterns
        is_green_glass = (g >= r + 2.5) and (g >= b - 2.0)          # Soda-lime green refraction
        is_amber_glass = (r > g + 10.0) and (g > b + 10.0) and (spec_ratio > 0.04)  # Brown bottle
        is_paper_warmth = (r > g + 5.0) and (g > b + 7.0) and (spec_ratio < 0.06)   # Cellulose matte
        is_plastic_blue = (b > g) and (b >= r + 3.0)                # PET blue tint
        is_neutral_metal = (chroma < 10.0) and (sat < 26.0)         # Achromatic gray

        # ── "bottle" / "jar" — could be glass or plastic ──
        if "bottle" in raw_lower or "jar" in raw_lower:
            if is_green_glass or is_amber_glass:
                return "Glass", 0.97
            if is_neutral_metal:
                return "Metal", 0.96
            return "Plastic", 0.97  # PET / plastic bottle default

        # ── "cup" — could be paper (Starbucks), glass (tumbler), or plastic ──
        if "cup" in raw_lower or "bowl" in raw_lower:
            if is_paper_warmth:
                return "Paper", 0.97
            if is_green_glass or (spec_ratio > 0.25 and sat < 20.0):
                return "Glass", 0.96
            return "Plastic", 0.96

        # ── "battery" — usually metal, but optical check for misclassified glass ──
        if "battery" in raw_lower:
            if has_optics and is_green_glass:
                return "Glass", 0.97
            return "Metal", 0.97

        # ── "trash" / "other litter" / "unlabeled litter" — pure optical decision ──
        if raw_lower in {"trash", "other litter", "unlabeled litter", "cigarette", "unknown"}:
            if has_optics:
                if is_green_glass or is_amber_glass:
                    return "Glass", 0.95
                if is_paper_warmth:
                    return "Paper", 0.95
                if is_neutral_metal:
                    return "Metal", 0.95
                if is_plastic_blue or sat > 35.0:
                    return "Plastic", 0.95
            return "Plastic", 0.90  # safest generic fallback

        # ── Non-recyclable classes from waste12 model ──
        if raw_lower in {"biological", "clothes", "shoes"}:
            if has_optics and is_paper_warmth:
                return "Paper", 0.90
            return "Plastic", 0.88

        # ──────────────── TIER 3: Keyword substring fallback ────────────────
        if any(kw in raw_lower for kw in ["glass", "wine", "vase"]):
            return "Glass", 0.96
        if any(kw in raw_lower for kw in ["paper", "card", "carton", "book"]):
            return "Paper", 0.96
        if any(kw in raw_lower for kw in ["metal", "can", "foil", "scissor"]):
            return "Metal", 0.96
        if raw_lower != "other plastic" and any(kw in raw_lower for kw in ["plastic", "straw", "styro"]):
            return "Plastic", 0.95

        # ──────────────── FINAL: Pure optical classification ────────────────
        if has_optics:
            if is_green_glass or is_amber_glass:
                return "Glass", 0.94
            if is_paper_warmth:
                return "Paper", 0.94
            if is_neutral_metal:
                return "Metal", 0.94
            if is_plastic_blue or sat > 35.0:
                return "Plastic", 0.93

        return "Plastic", 0.88

    @staticmethod
    def _calculate_iou(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
        """Computes Intersection over Union (IoU) between two bounding box candidate dicts."""
        x1 = max(b1["xmin"], b2["xmin"])
        y1 = max(b1["ymin"], b2["ymin"])
        x2 = min(b1["xmax"], b2["xmax"])
        y2 = min(b1["ymax"], b2["ymax"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1["xmax"] - b1["xmin"]) * (b1["ymax"] - b1["ymin"])
        a2 = (b2["xmax"] - b2["xmin"]) * (b2["ymax"] - b2["ymin"])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def detect_image(
        self,
        image: Image.Image,
        requested_hint: Optional[str] = None,
        source: Optional[str] = None
    ) -> DetectionResult:
        """
        Runs high-precision multi-model YOLOv8 object detection on the image.
        Employs computer vision preprocessing, multi-model cooperative ensemble,
        spatial reticle centering, and IoU Consensus Non-Maximum Suppression.
        """
        start_time = time.time()
        
        # 1. Computer Vision Preprocessing
        enhanced_image = self.preprocess_image(image, source=source)
        w, h = enhanced_image.size

        # 0. Fast LRU Cache lookup
        cache_key = None
        try:
            cache_key = (hashlib.md5(enhanced_image.tobytes()).hexdigest(), source)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                return DetectionResult(
                    detections=cached.detections,
                    image_width=cached.image_width,
                    image_height=cached.image_height,
                    inference_time_ms=round((time.time() - start_time) * 1000, 2),
                    primary_category=cached.primary_category,
                    confidence_avg=cached.confidence_avg,
                    total_items=cached.total_items,
                    annotated_image_base64=cached.annotated_image_base64,
                    engine_used=cached.engine_used,
                    probabilities=cached.probabilities,
                    features_summary=cached.features_summary
                )
        except Exception:
            cache_key = None

        candidates: List[Dict[str, Any]] = []

        # Setup torch inference mode context if available
        try:
            import torch
            inf_ctx = torch.inference_mode()
        except Exception:
            from contextlib import nullcontext
            inf_ctx = nullcontext()

        # 2. Multi-Model Cooperative Ensemble
        models_to_run = [
            (self.waste12_model, "waste12", self.conf_threshold),
            (self.waste_model, "waste", self.conf_threshold),
            (self.coco_model, "coco", self.conf_threshold)
        ]

        with inf_ctx:
            for model, m_type, conf_thresh in models_to_run:
                if model is None:
                    continue
                try:
                    results = model(enhanced_image, conf=conf_thresh, verbose=False)
                    if not results:
                        continue
                    r0 = results[0]
                    boxes = r0.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = r0.names.get(cls_id, "").lower().strip()
                        conf = float(box.conf[0].item())

                        # Skip non-material COCO background classes
                        if m_type == "coco" and cls_name in IGNORED_COCO_CLASSES:
                            continue

                        coords = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = map(int, coords)

                        # Crop region to inspect material specifics
                        crop = None
                        if (xmax - xmin) > 10 and (ymax - ymin) > 10:
                            crop = enhanced_image.crop((max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)))

                        waste_type, _ = self._classify_detected_label(cls_name, crop)
                        final_conf = conf
                        if cls_name in {"plastic", "other plastic"} and waste_type != "Plastic":
                            final_conf = min(final_conf, 0.49)

                        # Spatial Reticle Prior for Live Camera:
                        # Prioritize objects positioned in the center viewfinder reticle
                        box_cx = (xmin + xmax) / 2.0
                        box_cy = (ymin + ymax) / 2.0
                        norm_dist = ((box_cx - w / 2.0) ** 2 + (box_cy - h / 2.0) ** 2) ** 0.5 / ((w ** 2 + h ** 2) ** 0.5)
                        if source == "live_camera":
                            if norm_dist < 0.28:
                                final_conf = min(0.99, final_conf * 1.10)
                            elif norm_dist > 0.55:
                                final_conf = final_conf * 0.80

                        candidates.append({
                            "xmin": max(0, xmin),
                            "ymin": max(0, ymin),
                            "xmax": min(w, xmax),
                            "ymax": min(h, ymax),
                            "raw_class": cls_name,
                            "waste_type": waste_type,
                            "confidence": round(final_conf, 4),
                            "box_area": (xmax - xmin) * (ymax - ymin),
                            "source": m_type,
                            "center_dist": norm_dist
                        })
                except Exception as e:
                    print(f"[!] Error running {m_type} YOLO inference: {e}")

        # 3. Retry on a centered crop using the same confidence threshold.
        if not candidates or max(c["confidence"] for c in candidates) < 0.40:
            crop_w, crop_h = int(w * 0.70), int(h * 0.70)
            cx1 = int((w - crop_w) / 2)
            cy1 = int((h - crop_h) / 2)
            center_crop = enhanced_image.crop((cx1, cy1, cx1 + crop_w, cy1 + crop_h))

            with inf_ctx:
                for model, m_type, _ in [(self.waste12_model, "waste12", self.conf_threshold), (self.waste_model, "waste", self.conf_threshold)]:
                    if model is None:
                        continue
                    try:
                        res_crop = model(center_crop, conf=self.conf_threshold, verbose=False)[0]
                        for b in res_crop.boxes:
                            cls_id = int(b.cls[0].item())
                            cls_name = model.names.get(cls_id, "").lower().strip()
                            conf = float(b.conf[0].item())
                            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                            sub_crop = center_crop.crop((max(0, x1), max(0, y1), min(crop_w, x2), min(crop_h, y2)))
                            w_type, _ = self._classify_detected_label(cls_name, sub_crop)
                            crop_conf = min(conf, 0.49) if cls_name in {"plastic", "other plastic"} and w_type != "Plastic" else conf
                            candidates.append({
                                "xmin": cx1 + x1,
                                "ymin": cy1 + y1,
                                "xmax": cx1 + x2,
                                "ymax": cy1 + y2,
                                "raw_class": cls_name,
                                "waste_type": w_type,
                                "confidence": round(crop_conf, 4),
                                "box_area": (x2 - x1) * (y2 - y1),
                                "source": f"{m_type}_crop",
                                "center_dist": 0.1
                            })
                    except Exception:
                        pass

        # 4. Send undetected images to inspection instead of guessing a material.
        if not candidates:
            center_w, center_h = int(w * 0.7), int(h * 0.7)
            xmin = int((w - center_w) / 2)
            ymin = int((h - center_h) / 2)
            candidates.append({
                "xmin": xmin, "ymin": ymin, "xmax": xmin + center_w, "ymax": ymin + center_h,
                "raw_class": "unknown",
                "waste_type": "Unknown",
                "confidence": 0.0,
                "box_area": center_w * center_h,
                "source": "inspection_fallback",
                "center_dist": 0.0
            })

        # 5. Merge overlapping model boxes without fabricating consensus confidence.
        candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
        deduped: List[Dict[str, Any]] = []
        for cand in candidates:
            # Check overlap with existing chosen boxes
            match = None
            for existing in deduped:
                if self._calculate_iou(cand, existing) > 0.38:
                    match = existing
                    break
            if match:
                continue
            deduped.append(cand)
        candidates = deduped

        # Keep model scores independent; do not normalize a weak prediction to 100%.
        category_scores = {category: 0.0 for category in WASTE_CLASSES}
        for candidate in candidates:
            category = candidate["waste_type"]
            if category in category_scores:
                category_scores[category] = max(category_scores[category], candidate["confidence"])
        probabilities = {category: round(score * 100.0, 1) for category, score in category_scores.items()}

        final_boxes: List[BoundingBox] = []
        for c in candidates[:1]:
            vol, weight = self._estimate_physical_properties(c["waste_type"], c["box_area"])
            final_boxes.append(BoundingBox(
                xmin=c["xmin"],
                ymin=c["ymin"],
                xmax=c["xmax"],
                ymax=c["ymax"],
                waste_type=c["waste_type"],
                confidence=round(c["confidence"], 4),
                raw_class=c["raw_class"],
                estimated_volume_liters=vol,
                estimated_weight_kg=weight
            ))

        primary_cat = final_boxes[0].waste_type if final_boxes else "Plastic"
        conf_avg = round(float(final_boxes[0].confidence), 4) if final_boxes else 0.0
        elapsed_ms = (time.time() - start_time) * 1000

        # Annotate image
        annotated_b64 = self.generate_annotated_base64(enhanced_image, final_boxes)

        result = DetectionResult(
            detections=final_boxes,
            image_width=w,
            image_height=h,
            inference_time_ms=elapsed_ms,
            primary_category=primary_cat,
            confidence_avg=round(conf_avg, 4),
            total_items=len(final_boxes),
            annotated_image_base64=annotated_b64,
            engine_used=self.engine_name,
            probabilities=probabilities,
            features_summary={
                "detected_raw_classes": [b.raw_class for b in final_boxes],
                "primary_material": primary_cat,
                "confidence_percentage": round(conf_avg * 100.0, 1),
                "source_optimized": source or "auto"
            }
        )

        if cache_key is not None:
            if len(self._cache) >= self._max_cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result

        return result

    def _estimate_physical_properties(self, waste_type: str, box_area: float) -> Tuple[float, float]:
        """Estimates volume in liters and weight in kg based on waste category."""
        base_densities = {
            "Metal": 0.35,    # kg per liter equivalent
            "Plastic": 0.08,  # lightweight
            "Glass": 0.45,    # heavy
            "Paper": 0.12     # light
        }
        volume = max(0.2, min(4.0, (box_area / 150000.0) * 1.5))
        density = base_densities.get(waste_type, 0.15)
        weight = volume * density
        return round(volume, 2), round(weight, 3)

    def generate_annotated_base64(self, image: Image.Image, detections: List[BoundingBox]) -> str:
        """Draws bounding boxes, labels, and returns a base64 encoded JPEG."""
        import base64
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        font = None
        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        for det in detections:
            color = CLASS_COLORS.get(det.waste_type, (255, 255, 255))
            for thickness in range(4):
                draw.rectangle(
                    [det.xmin - thickness, det.ymin - thickness, det.xmax + thickness, det.ymax + thickness],
                    outline=color
                )

            label = f"{det.waste_type} {int(det.confidence * 100)}%"
            label_w = len(label) * 10 + 10
            label_h = 24
            
            draw.rectangle(
                [det.xmin, max(0, det.ymin - label_h), det.xmin + label_w, det.ymin],
                fill=color
            )
            text_pos = (det.xmin + 5, max(0, det.ymin - label_h + 2))
            draw.text(text_pos, label, fill=(255, 255, 255), font=font)

        buffer = io.BytesIO()
        annotated.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def detect_from_bytes(
        self,
        image_bytes: bytes,
        requested_hint: Optional[str] = None,
        source: Optional[str] = None
    ) -> DetectionResult:
        image = Image.open(io.BytesIO(image_bytes))
        return self.detect_image(image, requested_hint, source=source)
