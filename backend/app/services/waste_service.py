"""
Waste Service Facade
Integrates YOLO Detection, DSS Sorting Engine, Fill Tracker, and Route Optimizer.
Provides a centralized singleton service for FastAPI routes and frontend consumers.
"""

from datetime import datetime, timezone
import os
import random
from typing import Dict, Any, List, Optional
from PIL import Image
import io

from models.yolo_detector import WasteYOLODetector, DetectionResult
from backend.app.dss.sorting_engine import SortingEngineDSS
from backend.app.dss.fill_tracker import ContainerFillTracker
from backend.app.dss.route_optimizer import RouteOptimizerDSS
from backend.app.config import settings

class SmartWasteService:
    """
    Central orchestration service managing YOLO inference,
    DSS decision pipelines, telemetry tracking, and route analytics.
    """

    def __init__(self):
        self.detector = WasteYOLODetector(conf_threshold=0.45)
        self.sorting_engine = SortingEngineDSS(min_confidence=settings.MIN_CONFIDENCE_FOR_AUTOMATION)
        self.fill_tracker = ContainerFillTracker(
            warning_threshold=settings.WARNING_THRESHOLD_PCT,
            critical_threshold=settings.CRITICAL_THRESHOLD_PCT
        )
        self.route_optimizer = RouteOptimizerDSS()

        # Telemetry & processing historical counters
        self.total_processed_items = 428
        self.category_counts = {
            "Metal": 85,
            "Plastic": 164,
            "Glass": 72,
            "Paper": 107
        }
        self.total_weight_kg = 148.5
        self.total_volume_liters = 540.0
        self.successful_sorts = 419
        self.recent_decisions: List[Dict[str, Any]] = []

    def classify_and_process_image(
        self,
        image_bytes: bytes,
        category_hint: Optional[str] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full end-to-end processing pipeline:
        1. Run YOLO detection on uploaded image.
        2. For each detected object, evaluate DSS automated sorting rules.
        3. Update container fill levels and telemetry.
        4. Check and return system alerts.
        """
        # 1. YOLO Inference
        image = Image.open(io.BytesIO(image_bytes))
        detection_result: DetectionResult = self.detector.detect_image(
            image,
            requested_hint=category_hint,
            source=source
        )

        sorting_decisions = []
        updated_containers = []

        # 2. Process each detection through DSS
        for box in detection_result.detections:
            w_type = box.waste_type
            conf = box.confidence
            vol = box.estimated_volume_liters
            weight = box.estimated_weight_kg

            # Get target container to check current capacity
            target_bin = self.fill_tracker.get_primary_bin_for_type(w_type)
            current_fill_pct = target_bin["fill_percentage"] if target_bin else 50.0

            # DSS Automated Decision
            decision = self.sorting_engine.evaluate_item(
                detected_type=w_type,
                confidence=conf,
                target_container_fill_pct=current_fill_pct,
                weight_kg=weight,
                volume_liters=vol
            )
            sorting_decisions.append(decision)

            # 3. Update telemetry if accepted or routed to overflow
            if decision["status"] in ["ACCEPTED", "REROUTED_OVERFLOW"]:
                bin_id = decision["target_bin_id"]
                updated_bin = self.fill_tracker.record_deposit(bin_id, vol, weight)
                if updated_bin:
                    updated_containers.append(updated_bin)

                self.category_counts[w_type] = self.category_counts.get(w_type, 0) + 1
                self.total_processed_items += 1
                self.total_weight_kg += weight
                self.total_volume_liters += vol
                self.successful_sorts += 1
            else:
                self.total_processed_items += 1

            self.recent_decisions.insert(0, decision)
            if len(self.recent_decisions) > 50:
                self.recent_decisions.pop()

        # 4. Gather alerts
        alerts = self.fill_tracker.get_system_alerts()

        return {
            "detection": detection_result.to_dict(),
            "sorting_decisions": sorting_decisions,
            "updated_containers": updated_containers,
            "alerts_triggered": alerts
        }

    def process_manual_item(
        self,
        waste_type: str,
        confidence: float = 0.95,
        weight_kg: float = 0.2,
        volume_liters: float = 0.5
    ) -> Dict[str, Any]:
        """Processes a single item entry without image file."""
        target_bin = self.fill_tracker.get_primary_bin_for_type(waste_type)
        current_fill_pct = target_bin["fill_percentage"] if target_bin else 50.0

        decision = self.sorting_engine.evaluate_item(
            detected_type=waste_type,
            confidence=confidence,
            target_container_fill_pct=current_fill_pct,
            weight_kg=weight_kg,
            volume_liters=volume_liters
        )

        updated_bin = None
        if decision["status"] in ["ACCEPTED", "REROUTED_OVERFLOW"]:
            bin_id = decision["target_bin_id"]
            updated_bin = self.fill_tracker.record_deposit(bin_id, volume_liters, weight_kg)
            self.category_counts[waste_type] = self.category_counts.get(waste_type, 0) + 1
            self.total_processed_items += 1
            self.total_weight_kg += weight_kg
            self.total_volume_liters += volume_liters
            self.successful_sorts += 1
        else:
            self.total_processed_items += 1

        self.recent_decisions.insert(0, decision)
        alerts = self.fill_tracker.get_system_alerts()

        return {
            "decision": decision,
            "updated_container": updated_bin,
            "alerts": alerts
        }

    def get_containers(self) -> List[Dict[str, Any]]:
        return self.fill_tracker.get_all_containers()

    def update_container_status(self, bin_id: str, current_volume: Optional[float] = None, reset: bool = False):
        if reset:
            return self.fill_tracker.empty_container(bin_id)
        elif current_volume is not None:
            return self.fill_tracker.set_container_volume(bin_id, current_volume)
        return self.fill_tracker.get_container(bin_id)

    def optimize_route(self, min_fill_pct: float = 70.0) -> Dict[str, Any]:
        bins = self.fill_tracker.get_all_containers()
        return self.route_optimizer.optimize_collection_route(bins, min_fill_pct=min_fill_pct)

    def trigger_city_simulation_step(self) -> Dict[str, Any]:
        """Simulates 1 hour of smart city waste accumulation."""
        self.fill_tracker.simulate_step()
        # Randomly increment items
        added_type = random.choice(["Metal", "Plastic", "Glass", "Paper"])
        self.category_counts[added_type] += random.randint(1, 4)
        self.total_processed_items += random.randint(1, 4)
        return {
            "containers": self.fill_tracker.get_all_containers(),
            "alerts": self.fill_tracker.get_system_alerts()
        }

    def get_statistics_summary(self) -> Dict[str, Any]:
        containers = self.fill_tracker.get_all_containers()
        avg_fill = sum(c["fill_percentage"] for c in containers) / max(1, len(containers))

        critical_count = sum(1 for c in containers if c["status"] in ["CRITICAL", "OVERFLOW"])
        warning_count = sum(1 for c in containers if c["status"] == "WARNING")
        normal_count = sum(1 for c in containers if c["status"] == "NORMAL")

        total_cat = sum(self.category_counts.values())
        percentages = {
            k: round((v / total_cat) * 100.0, 1) if total_cat > 0 else 0.0
            for k, v in self.category_counts.items()
        }

        accuracy = round((self.successful_sorts / max(1, self.total_processed_items)) * 100.0, 1)
        carbon_offset = round(self.total_weight_kg * 1.85, 2) # 1.85kg CO2 saved per kg of recycled material

        return {
            "total_items_processed": self.total_processed_items,
            "category_counts": self.category_counts,
            "category_percentages": percentages,
            "total_weight_kg": round(self.total_weight_kg, 2),
            "total_volume_liters": round(self.total_volume_liters, 1),
            "overall_fill_average_pct": round(avg_fill, 1),
            "critical_bins_count": critical_count,
            "warning_bins_count": warning_count,
            "normal_bins_count": normal_count,
            "sorting_accuracy_rate_pct": accuracy,
            "carbon_offset_kg": carbon_offset,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def scan_mobile_image(
        self,
        image_bytes: bytes,
        category_hint: Optional[str] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Specialized endpoint for Mobile Smart Waste AI Scanner.
        Returns the exact Arabic attributes, material percentages,
        and DSS target bin recommendation matching the mobile UI design.
        """
        image = Image.open(io.BytesIO(image_bytes))
        det_result = self.detector.detect_image(
            image,
            requested_hint=category_hint,
            source=source
        )

        primary_cat = det_result.primary_category
        confidence_pct = round(det_result.confidence_avg * 100.0, 1)

        # Evaluate DSS mobile response with extracted probabilities
        mobile_result = self.sorting_engine.evaluate_mobile_scan(
            detected_category=primary_cat,
            overall_confidence=confidence_pct,
            custom_probabilities=det_result.probabilities if det_result.probabilities else None
        )

        mobile_result["annotated_image_base64"] = det_result.annotated_image_base64
        mobile_result["inference_time_ms"] = round(det_result.inference_time_ms, 1)

        # Update stats
        self.total_processed_items += 1
        if primary_cat in settings.CATEGORIES:
            self.category_counts[primary_cat] = self.category_counts.get(primary_cat, 0) + 1
        self.total_weight_kg += 0.25
        self.total_volume_liters += 0.55

        return mobile_result

# Singleton instance
waste_service = SmartWasteService()

