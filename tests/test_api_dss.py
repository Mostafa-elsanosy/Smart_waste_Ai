"""
Unit & Integration Tests for Smart Waste AI & DSS Backend
"""

import pytest
from fastapi.testclient import TestClient
import os
import io
from PIL import Image

from backend.app.main import app
from backend.app.dss.sorting_engine import SortingEngineDSS
from backend.app.dss.fill_tracker import ContainerFillTracker
from backend.app.dss.route_optimizer import RouteOptimizerDSS

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "yolo_detector_engine" in data
    assert data["dss_sorting_engine"] == "active"

def test_containers_endpoint():
    response = client.get("/api/containers")
    assert response.status_code == 200
    containers = response.json()
    assert len(containers) >= 4
    # Check required fields
    for c in containers:
        assert "id" in c
        assert "waste_type" in c
        assert "fill_percentage" in c
        assert "status" in c

def test_sorting_engine_dss_rules():
    engine = SortingEngineDSS(min_confidence=0.60)
    
    # 1. High confidence Metal sort
    decision_metal = engine.evaluate_item(
        detected_type="Metal",
        confidence=0.92,
        target_container_fill_pct=40.0
    )
    assert decision_metal["status"] == "ACCEPTED"
    assert "BIN-METAL" in decision_metal["target_bin_id"]
    assert "Pneumatic" in decision_metal["actuation_mechanism"]

    # 2. Low confidence -> Manual Inspection
    decision_low_conf = engine.evaluate_item(
        detected_type="Plastic",
        confidence=0.45,
        target_container_fill_pct=50.0
    )
    assert decision_low_conf["status"] == "REJECTED_INSPECTION"
    assert "INSPECTION" in decision_low_conf["target_bin_id"]

    # 3. High fill level (>95%) -> Reroute Overflow
    decision_overflow = engine.evaluate_item(
        detected_type="Glass",
        confidence=0.88,
        target_container_fill_pct=96.5
    )
    assert decision_overflow["status"] == "REROUTED_OVERFLOW"
    assert "OVERFLOW" in decision_overflow["target_bin_id"]

def test_route_optimizer_dss():
    optimizer = RouteOptimizerDSS()
    mock_bins = [
        {
            "id": "BIN-PLASTIC-01",
            "name": "Plastic Bin 1",
            "waste_type": "Plastic",
            "fill_percentage": 92.0,
            "status": "CRITICAL",
            "latitude": 40.7282,
            "longitude": -73.9942,
            "current_volume_liters": 198.0,
            "current_weight_kg": 15.8
        },
        {
            "id": "BIN-GLASS-01",
            "name": "Glass Bin 1",
            "waste_type": "Glass",
            "fill_percentage": 82.0,
            "status": "WARNING",
            "latitude": 40.7484,
            "longitude": -73.9857,
            "current_volume_liters": 142.0,
            "current_weight_kg": 64.2
        }
    ]
    route = optimizer.optimize_collection_route(mock_bins, min_fill_pct=70.0)
    assert route["total_bins_visited"] == 2
    assert route["total_distance_km"] > 0
    assert route["fuel_saved_liters"] >= 0
    assert route["co2_reduction_kg"] >= 0
    assert len(route["waypoints"]) == 2

def test_classify_endpoint_with_image():
    # Create simple dummy test image
    img = Image.new("RGB", (320, 240), color=(180, 200, 220))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/waste/classify",
        files={"file": ("test.jpg", buffer, "image/jpeg")},
        data={"category_hint": "Metal"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detection" in data
    assert "sorting_decisions" in data
    assert len(data["sorting_decisions"]) >= 1
    assert data["detection"]["primary_category"] in ["Metal", "Plastic", "Glass", "Paper"]

def test_manual_waste_entry():
    response = client.post("/api/waste/manual-entry?waste_type=Plastic&confidence=0.88&weight_kg=0.2&volume_liters=0.5")
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"]["detected_type"] == "Plastic"

def test_statistics_endpoint():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    stats = response.json()
    assert "total_items_processed" in stats
    assert "category_counts" in stats
    assert "carbon_offset_kg" in stats

def test_mobile_scan_endpoint():
    img = Image.new("RGB", (320, 240), color=(240, 230, 220))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/scan",
        files={"file": ("test_paper.jpg", buffer, "image/jpeg")},
        data={"category_hint": "Paper"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["detected_category"] == "Paper"
    assert data["title_ar"] == "منتج ورقي / كرتون (Paper / Cardboard)"
    assert data["status_badge"] == "قابل للتدوير"
    assert "dss_recommendation" in data
    assert "السلة الزرقاء" in data["dss_recommendation"]["target_bin_ar"]
    assert "probability_breakdown" in data
    assert len(data["probability_breakdown"]) == 4

def test_mobile_categories_and_stats():
    cat_res = client.get("/api/v1/categories")
    assert cat_res.status_code == 200
    categories = cat_res.json()
    assert len(categories) == 4
    cat_names = [c["name"] for c in categories]
    assert "Paper" in cat_names
    assert "Metal" in cat_names

    stats_res = client.get("/api/v1/mobile/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "scanned_items_count" in stats_data
    assert "co2_saved_kg" in stats_data

def test_feature_extracted_material_classification():
    from models.yolo_detector import WasteYOLODetector
    detector = WasteYOLODetector()
    
    # 1. Transparent Water Bottle -> Must be Plastic (Not Metal!)
    img_plastic = Image.open("data/sample_images/sample_plastic.jpg")
    res_plastic = detector.detect_image(img_plastic)
    assert res_plastic.primary_category == "Plastic", f"Expected Plastic, got {res_plastic.primary_category}"
    assert res_plastic.probabilities["Plastic"] > res_plastic.probabilities["Metal"]
    assert round(sum(res_plastic.probabilities.values()), 1) == 100.0

    # 2. Paper / Cardboard -> Must be Paper
    img_paper = Image.open("data/sample_images/sample_paper.jpg")
    res_paper = detector.detect_image(img_paper)
    assert res_paper.primary_category == "Paper", f"Expected Paper, got {res_paper.primary_category}"

    # 3. Glass Preserve Jar -> Must be Glass
    img_glass = Image.open("data/sample_images/sample_glass.jpg")
    res_glass = detector.detect_image(img_glass)
    assert res_glass.primary_category == "Glass", f"Expected Glass, got {res_glass.primary_category}"

    # 4. Metal Can -> Must be Metal
    img_metal = Image.open("data/sample_images/sample_metal.jpg")
    res_metal = detector.detect_image(img_metal)
    assert res_metal.primary_category == "Metal", f"Expected Metal, got {res_metal.primary_category}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])


