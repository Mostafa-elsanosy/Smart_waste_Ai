"""
Decision Support System (DSS) - Container Fill-Level Tracker
Maintains real-time telemetry, volume accumulation, weight sensors,
and alert state machines for smart waste containers.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import copy

class ContainerFillTracker:
    """
    Manages intelligent telemetry and fill level analytics for smart waste bins.
    """

    def __init__(self, warning_threshold: float = 75.0, critical_threshold: float = 90.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._containers: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_containers()

    def _initialize_default_containers(self):
        """Initializes realistic smart city bin telemetry across key urban zones."""
        default_bins = [
            {
                "id": "BIN-METAL-01",
                "name": "Central Plaza Metal Sorter",
                "waste_type": "Metal",
                "capacity_liters": 150.0,
                "current_volume_liters": 68.0,
                "current_weight_kg": 24.5,
                "location_name": "Downtown Central Hub",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "last_emptied": "2026-10-20T08:00:00Z",
                "item_count": 85,
                "temperature_celsius": 21.0,
                "battery_level_pct": 94,
                "fill_rate_liters_per_hour": 1.8
            },
            {
                "id": "BIN-PLASTIC-01",
                "name": "Tech District Plastic Station",
                "waste_type": "Plastic",
                "capacity_liters": 220.0,
                "current_volume_liters": 198.0, # 90.0% -> Critical!
                "current_weight_kg": 15.8,
                "location_name": "Innovation Boulevard",
                "latitude": 40.7282,
                "longitude": -73.9942,
                "last_emptied": "2026-10-19T14:30:00Z",
                "item_count": 312,
                "temperature_celsius": 23.2,
                "battery_level_pct": 88,
                "fill_rate_liters_per_hour": 4.2
            },
            {
                "id": "BIN-GLASS-01",
                "name": "Riverside Glass Vault",
                "waste_type": "Glass",
                "capacity_liters": 180.0,
                "current_volume_liters": 142.0, # 78.8% -> Warning!
                "current_weight_kg": 64.2,
                "location_name": "Riverside Marina Promenade",
                "latitude": 40.7484,
                "longitude": -73.9857,
                "last_emptied": "2026-10-18T10:15:00Z",
                "item_count": 120,
                "temperature_celsius": 19.8,
                "battery_level_pct": 91,
                "fill_rate_liters_per_hour": 2.1
            },
            {
                "id": "BIN-PAPER-01",
                "name": "University Campus Paper Compactor",
                "waste_type": "Paper",
                "capacity_liters": 300.0,
                "current_volume_liters": 125.0, # 41.6% -> Normal
                "current_weight_kg": 28.0,
                "location_name": "Campus Library Square",
                "latitude": 40.7306,
                "longitude": -73.9352,
                "last_emptied": "2026-10-21T06:00:00Z",
                "item_count": 210,
                "temperature_celsius": 20.5,
                "battery_level_pct": 97,
                "fill_rate_liters_per_hour": 3.0
            },
            {
                "id": "BIN-PLASTIC-02",
                "name": "Market Square Plastic Collector",
                "waste_type": "Plastic",
                "capacity_liters": 200.0,
                "current_volume_liters": 155.0, # 77.5% -> Warning!
                "current_weight_kg": 12.4,
                "location_name": "Old Market Square",
                "latitude": 40.7589,
                "longitude": -73.9851,
                "last_emptied": "2026-10-19T11:00:00Z",
                "item_count": 240,
                "temperature_celsius": 22.8,
                "battery_level_pct": 85,
                "fill_rate_liters_per_hour": 3.8
            },
            {
                "id": "BIN-METAL-02",
                "name": "Harbor Industrial Metal Bin",
                "waste_type": "Metal",
                "capacity_liters": 160.0,
                "current_volume_liters": 50.0, # 31.2% -> Normal
                "current_weight_kg": 19.5,
                "location_name": "South Ferry Docks",
                "latitude": 40.7015,
                "longitude": -74.0135,
                "last_emptied": "2026-10-20T16:00:00Z",
                "item_count": 62,
                "temperature_celsius": 18.5,
                "battery_level_pct": 99,
                "fill_rate_liters_per_hour": 1.2
            }
        ]

        for b in default_bins:
            self._update_status(b)
            self._containers[b["id"]] = b

    def _update_status(self, b: Dict[str, Any]):
        cap = max(1.0, b["capacity_liters"])
        vol = max(0.0, b["current_volume_liters"])
        pct = round((vol / cap) * 100.0, 1)
        b["fill_percentage"] = pct

        if pct >= 100.0:
            b["status"] = "OVERFLOW"
        elif pct >= self.critical_threshold:
            b["status"] = "CRITICAL"
        elif pct >= self.warning_threshold:
            b["status"] = "WARNING"
        else:
            b["status"] = "NORMAL"

        rate = b.get("fill_rate_liters_per_hour", 2.0)
        remaining_vol = max(0.0, cap - vol)
        b["estimated_hours_to_full"] = round(remaining_vol / rate, 1) if rate > 0 else 99.9

    def get_all_containers(self) -> List[Dict[str, Any]]:
        return list(self._containers.values())

    def get_container(self, bin_id: str) -> Optional[Dict[str, Any]]:
        return self._containers.get(bin_id)

    def get_primary_bin_for_type(self, waste_type: str) -> Optional[Dict[str, Any]]:
        """Returns default primary bin for a waste category."""
        for b in self._containers.values():
            if b["waste_type"] == waste_type:
                return b
        return None

    def record_deposit(self, bin_id: str, volume_liters: float, weight_kg: float) -> Optional[Dict[str, Any]]:
        """Adds deposited waste volume and weight to container."""
        b = self._containers.get(bin_id)
        if not b:
            # Check if there is a primary container for this type
            for container in self._containers.values():
                if container["waste_type"].lower() in bin_id.lower():
                    b = container
                    break
        if not b:
            # Fallback to first container
            b = list(self._containers.values())[0]

        b["current_volume_liters"] = min(b["capacity_liters"] * 1.05, b["current_volume_liters"] + volume_liters)
        b["current_weight_kg"] += weight_kg
        b["item_count"] += 1
        self._update_status(b)
        return copy.deepcopy(b)

    def empty_container(self, bin_id: str) -> Optional[Dict[str, Any]]:
        """Resets container fill level after truck collection."""
        b = self._containers.get(bin_id)
        if b:
            b["current_volume_liters"] = 0.0
            b["current_weight_kg"] = 0.0
            b["item_count"] = 0
            b["last_emptied"] = datetime.now(timezone.utc).isoformat()
            self._update_status(b)
            return copy.deepcopy(b)
        return None

    def empty_all_containers(self) -> List[Dict[str, Any]]:
        for b in self._containers.values():
            b["current_volume_liters"] = 0.0
            b["current_weight_kg"] = 0.0
            b["item_count"] = 0
            b["last_emptied"] = datetime.now(timezone.utc).isoformat()
            self._update_status(b)
        return self.get_all_containers()

    def set_container_volume(self, bin_id: str, volume_liters: float) -> Optional[Dict[str, Any]]:
        b = self._containers.get(bin_id)
        if b:
            b["current_volume_liters"] = max(0.0, min(b["capacity_liters"] * 1.1, volume_liters))
            self._update_status(b)
            return copy.deepcopy(b)
        return None

    def simulate_step(self) -> List[Dict[str, Any]]:
        """Simulates gradual urban waste accumulation."""
        import random
        for b in self._containers.values():
            delta = random.uniform(1.0, 4.5)
            b["current_volume_liters"] = min(b["capacity_liters"], b["current_volume_liters"] + delta)
            b["current_weight_kg"] += round(delta * random.uniform(0.1, 0.3), 2)
            self._update_status(b)
        return self.get_all_containers()

    def get_critical_and_warning_bins(self) -> List[Dict[str, Any]]:
        """Returns bins that require servicing (CRITICAL or WARNING)."""
        return [b for b in self._containers.values() if b["status"] in ["CRITICAL", "WARNING", "OVERFLOW"]]

    def get_system_alerts(self) -> List[str]:
        alerts = []
        for b in self._containers.values():
            if b["status"] == "OVERFLOW":
                alerts.append(f"EMERGENCY OVERFLOW: {b['name']} ({b['id']}) is at {b['fill_percentage']}% capacity!")
            elif b["status"] == "CRITICAL":
                alerts.append(f"CRITICAL ALERT: {b['name']} ({b['id']}) reached {b['fill_percentage']}% fill level. Collection needed immediately.")
            elif b["status"] == "WARNING":
                alerts.append(f"WARNING: {b['name']} ({b['id']}) reached {b['fill_percentage']}% fill level.")
        return alerts
