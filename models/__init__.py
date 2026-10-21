"""
Smart Waste AI Models Package
Contains YOLO detection engine with support for YOLOv8 and computer-vision enhanced simulation.
"""
from .yolo_detector import WasteYOLODetector, DetectionResult, BoundingBox

__all__ = ["WasteYOLODetector", "DetectionResult", "BoundingBox"]
