"""
Configuration settings for Smart Waste AI & Decision Support System (DSS)
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any, List

class SystemSettings(BaseSettings):
    APP_NAME: str = "Smart Waste AI & DSS"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Waste categories
    CATEGORIES: List[str] = ["Metal", "Plastic", "Glass", "Paper"]
    
    # Fill level alert thresholds (%)
    WARNING_THRESHOLD_PCT: float = 75.0
    CRITICAL_THRESHOLD_PCT: float = 90.0
    
    # Sorting confidence threshold
    MIN_CONFIDENCE_FOR_AUTOMATION: float = 0.60
    
    # Collection Fleet parameters
    TRUCK_CAPACITY_KG: float = 2500.0
    TRUCK_VOLUME_LITERS: float = 6000.0
    AVERAGE_SPEED_KMH: float = 35.0
    FUEL_CONSUMPTION_PER_100KM: float = 28.0 # Liters of diesel
    CO2_KG_PER_LITER_DIESEL: float = 2.68

settings = SystemSettings()
