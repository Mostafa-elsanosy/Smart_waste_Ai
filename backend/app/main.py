"""
FastAPI Main Application for Smart Waste AI & Decision Support System (DSS)
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from contextlib import asynccontextmanager
from PIL import Image
from backend.app.config import settings
from backend.app.services.waste_service import waste_service
from backend.app.schemas import (
    DetectionResponseSchema,
    ProcessDepositResponse,
    ContainerSchema,
    ContainerUpdateSchema,
    RouteOptimizationResponse,
    StatisticsSummarySchema,
    MobileScanResponse
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Pre-warms YOLO models to eliminate cold-start latency."""
    try:
        dummy = Image.new("RGB", (320, 240), (200, 200, 200))
        await run_in_threadpool(waste_service.detector.detect_image, dummy)
    except Exception:
        pass
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Smart Waste Management Backend with YOLO Detection and Decision Support System (DSS) Logic.",
    lifespan=lifespan
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(BASE_DIR, "frontend", "web")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "sample_images")
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit for safety

# Mount web frontend static assets if directory exists
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def root():
    """Serves the web dashboard or API overview."""
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "status": "operational"
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint indicating detector model and DSS status."""
    return {
        "status": "online",
        "version": settings.APP_VERSION,
        "yolo_detector_engine": waste_service.detector.engine_name,
        "has_ultralytics": waste_service.detector.yolo_model is not None,
        "dss_sorting_engine": "active",
        "dss_route_optimizer": "active",
        "total_containers_tracked": len(waste_service.get_containers())
    }


@app.post("/api/waste/classify", response_model=ProcessDepositResponse)
async def classify_waste_image(
    file: UploadFile = File(...),
    category_hint: Optional[str] = Form(None),
    source: Optional[str] = Form(None)
):
    """
    Classifies uploaded waste image via YOLOv8, executes DSS sorting rules,
    updates container fill tracking, and returns real-time decisions and annotated image.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large (maximum size is 15MB).")
        
        result = await run_in_threadpool(
            waste_service.classify_and_process_image,
            image_bytes,
            category_hint=category_hint,
            source=source
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")


@app.post("/api/waste/manual-entry")
def manual_waste_entry(
    waste_type: str = Query(..., description="One of: Metal, Plastic, Glass, Paper"),
    confidence: float = Query(0.95, ge=0.0, le=1.0),
    weight_kg: float = Query(0.25, ge=0.01),
    volume_liters: float = Query(0.6, ge=0.05)
):
    """Manual waste item entry to test DSS rules without uploading an image."""
    if waste_type not in settings.CATEGORIES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid waste type '{waste_type}'. Must be one of {settings.CATEGORIES}"
        )
    return waste_service.process_manual_item(waste_type, confidence, weight_kg, volume_liters)


@app.get("/api/containers", response_model=List[ContainerSchema])
def list_containers():
    """Returns real-time telemetry for all smart waste bins."""
    return waste_service.get_containers()


@app.get("/api/containers/{bin_id}", response_model=ContainerSchema)
def get_container(bin_id: str):
    """Returns telemetry for a specific container."""
    container = waste_service.fill_tracker.get_container(bin_id)
    if not container:
        raise HTTPException(status_code=404, detail=f"Container with ID '{bin_id}' not found.")
    return container


@app.put("/api/containers/{bin_id}")
def update_container(bin_id: str, payload: ContainerUpdateSchema):
    """Updates container volume or resets container to empty (collection completion)."""
    updated = waste_service.update_container_status(
        bin_id=bin_id,
        current_volume=payload.current_volume_liters,
        reset=payload.reset_to_empty or False
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Container '{bin_id}' not found.")
    return updated


@app.post("/api/containers/empty-all")
def empty_all_containers():
    """Resets all containers in the smart city network."""
    return waste_service.fill_tracker.empty_all_containers()


@app.get("/api/dss/route-optimization", response_model=RouteOptimizationResponse)
def get_route_optimization(min_fill_pct: float = Query(70.0, ge=10.0, le=100.0)):
    """
    DSS Logistics Engine:
    Solves Traveling Salesperson Problem (TSP) with Nearest Neighbor and 2-Opt local search
    to optimize collection trucks visiting critical/warning bins while computing carbon/fuel savings.
    """
    return waste_service.optimize_route(min_fill_pct=min_fill_pct)


@app.post("/api/simulation/tick")
def simulate_accumulation():
    """Simulates 1 hour of smart city waste accumulation across containers."""
    return waste_service.trigger_city_simulation_step()


@app.get("/api/statistics", response_model=StatisticsSummarySchema)
def get_statistics():
    """Returns aggregate analytics, category distributions, accuracy, and carbon offset."""
    return waste_service.get_statistics_summary()


@app.get("/api/sample-images/{filename}")
def get_sample_image(filename: str):
    """Serves generated sample waste images for instant testing in dashboard with path traversal protection."""
    safe_name = os.path.basename(filename)
    filepath = os.path.abspath(os.path.join(SAMPLES_DIR, safe_name))
    samples_dir_abs = os.path.abspath(SAMPLES_DIR)
    
    # Path traversal protection
    if not filepath.startswith(samples_dir_abs) or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(filepath)


@app.get("/api/sample-images")
def list_sample_images():
    """Lists available sample images for instant classification testing."""
    if not os.path.exists(SAMPLES_DIR):
        return []
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
    return [
        {
            "filename": f,
            "url": f"/api/sample-images/{f}",
            "waste_type": f.replace("sample_", "").split(".")[0].capitalize()
        }
        for f in files
    ]


# =====================================================================
# Mobile App Backend API v1 (Exact Match for Flutter & React Native UI)
# =====================================================================

@app.get("/mobile")
def mobile_view():
    """Serves the exact pixel-perfect mobile interactive simulator."""
    mobile_file = os.path.join(WEB_DIR, "mobile.html")
    if os.path.exists(mobile_file):
        return FileResponse(mobile_file)
    return {"message": "Mobile interface file not found."}


@app.post("/api/v1/scan", response_model=MobileScanResponse)
async def mobile_scan_endpoint(
    file: UploadFile = File(...),
    category_hint: Optional[str] = Form(None),
    source: Optional[str] = Form(None)
):
    """
    Core Mobile App Scan Endpoint.
    Accepts camera or gallery frame, executes YOLOv8 material classification,
    applies DSS rules, and returns Arabic labels, probabilities breakdown,
    and target bin color/advice matching the design specification.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image frame cannot be empty.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Image frame too large (maximum size is 15MB).")
        
        result = await run_in_threadpool(
            waste_service.scan_mobile_image,
            image_bytes,
            category_hint=category_hint,
            source=source
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan error: {str(e)}")


@app.get("/api/v1/categories")
def get_mobile_categories():
    """
    Returns the 4 waste categories matching the Home Screen 2x2 grid.
    Includes Arabic titles, bin descriptions, icons, and item counts.
    """
    stats = waste_service.get_statistics_summary()
    counts = stats.get("category_counts", {})

    return [
        {
            "id": "paper",
            "name": "Paper",
            "title_ar": "ورق وكرتون (Paper)",
            "bin_target_ar": "السلة الزرقاء (مخصصة للورق والكرتون)",
            "icon": "book",
            "count": counts.get("Paper", 0),
            "color": "#3B82F6",
            "bg_color": "#EFF6FF",
            "border_color": "#BFDBFE",
            "badge_color": "#2563EB"
        },
        {
            "id": "metal",
            "name": "Metal",
            "title_ar": "معادن (Metal)",
            "bin_target_ar": "السلة الصفراء / المعدنية",
            "icon": "can",
            "count": counts.get("Metal", 0),
            "color": "#F59E0B",
            "bg_color": "#FFFBEB",
            "border_color": "#FDE68A",
            "badge_color": "#D97706"
        },
        {
            "id": "plastic",
            "name": "Plastic",
            "title_ar": "بلاستيك (Plastic)",
            "bin_target_ar": "السلة البرتقالية / الحمراء (للبلاستيك)",
            "icon": "bottle",
            "count": counts.get("Plastic", 0),
            "color": "#EF4444",
            "bg_color": "#FEF2F2",
            "border_color": "#FECACA",
            "badge_color": "#DC2626"
        },
        {
            "id": "glass",
            "name": "Glass",
            "title_ar": "زجاج (Glass)",
            "bin_target_ar": "السلة الخضراء (مخصصة للزجاجيات)",
            "icon": "glass",
            "count": counts.get("Glass", 0),
            "color": "#10B981",
            "bg_color": "#ECFDF5",
            "border_color": "#A7F3D0",
            "badge_color": "#059669"
        }
    ]


@app.get("/api/v1/mobile/stats")
def get_mobile_stats():
    """
    Returns KPIs for the mobile Home Screen Hero:
    - Scanned items count (عنصر تم فحصه)
    - Saved CO2 in kg (CO2 تم توفيره)
    - Number of classified categories (أنواع مصنفة)
    """
    stats = waste_service.get_statistics_summary()
    return {
        "scanned_items_count": stats.get("total_items_processed", 0),
        "co2_saved_kg": stats.get("carbon_offset_kg", 0.0),
        "categories_count": 4,
        "recycling_rate_pct": stats.get("sorting_accuracy_rate_pct", 97.5)
    }

