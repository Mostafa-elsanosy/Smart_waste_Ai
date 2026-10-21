"""
Pydantic Schemas for Smart Waste AI & DSS APIs
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBoxSchema(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    waste_type: str
    confidence: float
    estimated_volume_liters: float
    estimated_weight_kg: float

class DetectionResponseSchema(BaseModel):
    primary_category: str
    confidence_avg: float
    total_items: int
    inference_time_ms: float
    image_width: int
    image_height: int
    engine_used: str
    detections: List[BoundingBoxSchema]
    annotated_image_base64: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None

class SortingDecisionSchema(BaseModel):
    item_id: str
    detected_type: str
    confidence: float
    target_bin_id: str
    target_bin_name: str
    gate_action: str
    actuation_mechanism: str
    conveyor_speed_mps: float
    status: str # "ACCEPTED", "REROUTED_OVERFLOW", "REJECTED_INSPECTION"
    decision_reasoning: List[str]
    timestamp: str

class ProcessDepositResponse(BaseModel):
    detection: DetectionResponseSchema
    sorting_decisions: List[SortingDecisionSchema]
    updated_containers: List[Dict[str, Any]]
    alerts_triggered: List[str]

class ContainerSchema(BaseModel):
    id: str
    name: str
    waste_type: str
    capacity_liters: float
    current_volume_liters: float
    current_weight_kg: float
    fill_percentage: float
    status: str # "NORMAL", "WARNING", "CRITICAL", "OVERFLOW"
    location_name: str
    latitude: float
    longitude: float
    last_emptied: str
    item_count: int
    temperature_celsius: float = 22.5
    battery_level_pct: int = 95

class ContainerUpdateSchema(BaseModel):
    current_volume_liters: Optional[float] = None
    reset_to_empty: Optional[bool] = False

class RouteWaypointSchema(BaseModel):
    stop_number: int
    bin_id: str
    bin_name: str
    waste_type: str
    fill_percentage: float
    volume_to_collect_liters: float
    weight_to_collect_kg: float
    latitude: float
    longitude: float
    cumulative_distance_km: float
    estimated_arrival_minutes: float
    urgency_level: str

class RouteOptimizationResponse(BaseModel):
    route_id: str
    timestamp: str
    total_bins_visited: int
    total_distance_km: float
    estimated_time_minutes: float
    total_volume_collected_liters: float
    total_weight_collected_kg: float
    fuel_saved_liters: float
    co2_reduction_kg: float
    algorithm_used: str
    waypoints: List[RouteWaypointSchema]

class StatisticsSummarySchema(BaseModel):
    total_items_processed: int
    category_counts: Dict[str, int]
    category_percentages: Dict[str, float]
    total_weight_kg: float
    total_volume_liters: float
    overall_fill_average_pct: float
    critical_bins_count: int
    warning_bins_count: int
    normal_bins_count: int
    sorting_accuracy_rate_pct: float
    carbon_offset_kg: float
    last_updated: str

class MaterialProbability(BaseModel):
    material: str
    material_ar: str
    percentage: float
    color: str
    is_primary: bool

class DSSMobileRecommendation(BaseModel):
    bin_label_ar: str = "حاوية المهملات المناسبة:"
    target_bin_ar: str
    bin_color_name: str
    bin_color_hex: str
    icon: str = "trash"
    advice_ar: str
    decision_status: str = "APPROVED"

class MobileScanResponse(BaseModel):
    status: str = "success"
    detected_category: str
    category_badge: str
    title_ar: str
    description_ar: str
    recyclable: bool = True
    status_badge: str = "قابل للتدوير"
    status_badge_color: str = "#10B981"
    overall_confidence: float
    probabilities: Dict[str, float]
    probability_breakdown: List[MaterialProbability]
    dss_recommendation: DSSMobileRecommendation
    annotated_image_base64: Optional[str] = None
    inference_time_ms: float = 18.5
