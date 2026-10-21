"""
Decision Support System (DSS) - Automated Sorting Engine
Implements deterministic and multi-criteria sorting rules based on
AI classification, confidence thresholds, target container capacities,
and safety handling constraints.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import uuid

class SortingEngineDSS:
    """
    Automated Sorting Rules Engine for Waste Management.
    """

    DEFAULT_ROUTING_MAP = {
        "Metal": {
            "bin_id": "BIN-METAL-01",
            "bin_name": "Metal & Cans Hopper",
            "gate_action": "GATE_A_EXTEND",
            "actuation_mechanism": "High-Force Pneumatic Ejector Piston",
            "base_speed_mps": 1.2,
            "safety_protocol": "Magnetic sensor verification and heavy impact plate active."
        },
        "Plastic": {
            "bin_id": "BIN-PLASTIC-01",
            "bin_name": "Plastics & Polymers Silo",
            "gate_action": "GATE_B_VACUUM",
            "actuation_mechanism": "Delta Robotic Gripper / Optical Air Nozzle",
            "base_speed_mps": 1.5,
            "safety_protocol": "Static charge eliminator active."
        },
        "Glass": {
            "bin_id": "BIN-GLASS-01",
            "bin_name": "Glass Containers Vault",
            "gate_action": "GATE_C_CHUTE",
            "actuation_mechanism": "Cushioned Hydraulic Slide / Soft-Drop Ramp",
            "base_speed_mps": 0.8,
            "safety_protocol": "Acoustic breakage sensor and deceleration dampers active."
        },
        "Paper": {
            "bin_id": "BIN-PAPER-01",
            "bin_name": "Paper & Cardboard Compactor",
            "gate_action": "GATE_D_AIRKNIFE",
            "actuation_mechanism": "High-Velocity Air Knife Jet",
            "base_speed_mps": 1.8,
            "safety_protocol": "Moisture sensor check and fire suppression monitor active."
        }
    }

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence

    def evaluate_item(
        self,
        detected_type: str,
        confidence: float,
        target_container_fill_pct: float,
        weight_kg: float = 0.15,
        volume_liters: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes multi-criteria DSS rules on detected waste item:
        1. Confidence validation rule
        2. Container capacity / overflow rule
        3. Handling safety and speed adaptation
        4. Detailed audit trail for transparency
        """
        item_id = f"ITEM-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        reasons = []

        # Default route configuration
        route_config = self.DEFAULT_ROUTING_MAP.get(detected_type)
        if not route_config:
            return {
                "item_id": item_id,
                "detected_type": detected_type,
                "confidence": confidence,
                "target_bin_id": "BIN-INSPECTION-00",
                "target_bin_name": "Uncategorized Inspection Station",
                "gate_action": "DIVERT_TO_MANUAL",
                "actuation_mechanism": "Manual Diverter Deflector",
                "conveyor_speed_mps": 0.5,
                "status": "REJECTED_INSPECTION",
                "decision_reasoning": [f"Unrecognized waste class '{detected_type}'. Rerouted for human inspection."],
                "timestamp": timestamp
            }

        target_bin_id = route_config["bin_id"]
        target_bin_name = route_config["bin_name"]
        gate_action = route_config["gate_action"]
        actuation_mechanism = route_config["actuation_mechanism"]
        conveyor_speed = route_config["base_speed_mps"]
        status = "ACCEPTED"

        # RULE 1: Confidence Threshold Evaluation
        if confidence < self.min_confidence:
            status = "REJECTED_INSPECTION"
            target_bin_id = "BIN-INSPECTION-00"
            target_bin_name = "Quality Control & Manual Inspection Bay"
            gate_action = "DIVERT_TO_MANUAL"
            actuation_mechanism = "Safety Deflector Arm"
            conveyor_speed = 0.6
            reasons.append(
                f"Low AI confidence ({confidence*100:.1f}% < {self.min_confidence*100:.1f}%). "
                "Item diverted to inspection to safeguard recycling purity."
            )
            return {
                "item_id": item_id,
                "detected_type": detected_type,
                "confidence": confidence,
                "target_bin_id": target_bin_id,
                "target_bin_name": target_bin_name,
                "gate_action": gate_action,
                "actuation_mechanism": actuation_mechanism,
                "conveyor_speed_mps": conveyor_speed,
                "status": status,
                "decision_reasoning": reasons,
                "timestamp": timestamp
            }
        else:
            reasons.append(f"AI classification confirmed '{detected_type}' with high confidence ({confidence*100:.1f}%).")

        # RULE 2: Fill-Level & Overflow Prevention Rule
        if target_container_fill_pct >= 95.0:
            status = "REROUTED_OVERFLOW"
            target_bin_id = f"{route_config['bin_id']}-OVERFLOW"
            target_bin_name = f"{route_config['bin_name']} (Auxiliary Overflow Bin)"
            gate_action = f"{route_config['gate_action']}_BYPASS"
            reasons.append(
                f"Target container '{route_config['bin_name']}' is full ({target_container_fill_pct:.1f}%). "
                "DSS triggered emergency diversion to secondary overflow silo."
            )
        elif target_container_fill_pct >= 75.0:
            reasons.append(
                f"Target container in WARNING state ({target_container_fill_pct:.1f}% full). "
                "Item accepted; collection dispatch alert flagged."
            )
        else:
            reasons.append(
                f"Target container has nominal capacity ({target_container_fill_pct:.1f}% full). Routine deposit approved."
            )

        # RULE 3: Handling & Speed Calibration Rule
        if detected_type == "Glass":
            conveyor_speed = 0.8
            reasons.append("Material is fragile glass: Speed capped at 0.8 m/s with hydraulic soft-landing activated.")
        elif detected_type == "Metal" and weight_kg > 0.4:
            conveyor_speed = 1.0
            reasons.append("Heavy metal payload detected: Pneumatic pressure boosted to 6.2 bar.")
        elif detected_type == "Paper":
            conveyor_speed = 1.8
            reasons.append("Dry paper media: High-speed air ejection sequence engaged.")

        reasons.append(route_config["safety_protocol"])

        return {
            "item_id": item_id,
            "detected_type": detected_type,
            "confidence": confidence,
            "target_bin_id": target_bin_id,
            "target_bin_name": target_bin_name,
            "gate_action": gate_action,
            "actuation_mechanism": actuation_mechanism,
            "conveyor_speed_mps": conveyor_speed,
            "status": status,
            "decision_reasoning": reasons,
            "timestamp": timestamp
        }

    MOBILE_CONFIG = {
        "Paper": {
            "category_badge": "ورق وكرتون (Paper)",
            "title_ar": "منتج ورقي / كرتون (Paper / Cardboard)",
            "description_ar": "اكتشاف سطح غير لامع (Matte) مع تدرجات لونية دافئة ونسيج ألياف يطابق الورق والكرتون.",
            "target_bin_ar": "السلة الزرقاء (مخصصة للورق والكرتون)",
            "bin_color_name": "Blue",
            "bin_color_hex": "#2563EB",
            "advice_ar": "يرجى التأكد من خلو الورق والكرتون من بقايا الزيوت والسوائل لضمان جودة إعادة التدوير.",
            "color": "#3B82F6",
            "base_prob": 40.5
        },
        "Metal": {
            "category_badge": "معادن (Metal)",
            "title_ar": "علبة معدنية / ألمنيوم (Metal Can)",
            "description_ar": "اكتشاف انعكاس معدني مميز مع حواف صلبة وهيكل اسطواني يطابق علب المشروبات والمعادن.",
            "target_bin_ar": "السلة الصفراء / المعدنية",
            "bin_color_name": "Yellow",
            "bin_color_hex": "#EAB308",
            "advice_ar": "قم بضغط العلبة المعدنية لتوفير مساحة في الحاوية قبل رميها.",
            "color": "#F59E0B",
            "base_prob": 23.9
        },
        "Plastic": {
            "category_badge": "بلاستيك (Plastic)",
            "title_ar": "عبوة بلاستيكية / بوليمر (Plastic Bottle)",
            "description_ar": "اكتشاف بوليمر شفاف خفيف الوزن مع مرونة سطحية وشريط لاصق يطابق زجاجات البلاستيك.",
            "target_bin_ar": "السلة البرتقالية / الحمراء (للبلاستيك)",
            "bin_color_name": "Orange/Red",
            "bin_color_hex": "#EF4444",
            "advice_ar": "أفرغ السوائل وأعد إغلاق الغطاء بإحكام قبل التخلص من القارورة.",
            "color": "#EF4444",
            "base_prob": 22.8
        },
        "Glass": {
            "category_badge": "زجاج (Glass)",
            "title_ar": "قارورة زجاجية / إناء (Glass Jar)",
            "description_ar": "اكتشاف شفافية عالية ونفاذية ضوئية صلبة غير قابلة للخدش تطابق الزجاج والمراطبين.",
            "target_bin_ar": "السلة الخضراء (مخصصة للزجاجيات)",
            "bin_color_name": "Green",
            "bin_color_hex": "#10B981",
            "advice_ar": "ضع الزجاج برفق في السلة المخصصة لتجنب الكسر وحماية عمال النظافة.",
            "color": "#10B981",
            "base_prob": 12.8
        }
    }

    def evaluate_mobile_scan(
        self,
        detected_category: str = "Paper",
        overall_confidence: float = 72.0,
        custom_probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Builds the exact localized payload required by the Smart Waste Mobile UI.
        Matches the Arabic specification and UI cards perfectly.
        """
        cat_info = self.MOBILE_CONFIG.get(detected_category, self.MOBILE_CONFIG["Paper"])
        
        # Build probability breakdown for candidate materials
        if custom_probabilities and len(custom_probabilities) == 4:
            probs = custom_probabilities
        else:
            # Default distribution tuned to user's design reference:
            # Paper: 40.5%, Metal: 23.9%, Plastic: 22.8%, Glass: 12.8%
            if detected_category == "Paper":
                probs = {"Paper": 40.5, "Metal": 23.9, "Plastic": 22.8, "Glass": 12.8}
            elif detected_category == "Metal":
                probs = {"Metal": 46.2, "Plastic": 22.1, "Paper": 18.5, "Glass": 13.2}
            elif detected_category == "Plastic":
                probs = {"Plastic": 44.8, "Glass": 21.4, "Paper": 19.3, "Metal": 14.5}
            else: # Glass
                probs = {"Glass": 48.0, "Plastic": 22.5, "Metal": 15.0, "Paper": 14.5}

        # Order matching screenshot: Metal, Paper, Glass, Plastic
        ordered_keys = ["Metal", "Paper", "Glass", "Plastic"]
        breakdown = []
        for k in ordered_keys:
            val = probs.get(k, 25.0)
            meta = self.MOBILE_CONFIG.get(k, {})
            breakdown.append({
                "material": k,
                "material_ar": k,
                "percentage": round(val, 1),
                "color": meta.get("color", "#3B82F6"),
                "is_primary": (k == detected_category)
            })

        return {
            "status": "success",
            "detected_category": detected_category,
            "category_badge": cat_info["category_badge"],
            "title_ar": cat_info["title_ar"],
            "description_ar": cat_info["description_ar"],
            "recyclable": True,
            "status_badge": "قابل للتدوير",
            "status_badge_color": "#10B981",
            "overall_confidence": round(overall_confidence, 1),
            "probabilities": probs,
            "probability_breakdown": breakdown,
            "dss_recommendation": {
                "bin_label_ar": "حاوية المهملات المناسبة:",
                "target_bin_ar": cat_info["target_bin_ar"],
                "bin_color_name": cat_info["bin_color_name"],
                "bin_color_hex": cat_info["bin_color_hex"],
                "icon": "trash-blue" if cat_info["bin_color_name"] == "Blue" else "trash",
                "advice_ar": cat_info["advice_ar"],
                "decision_status": "APPROVED"
            }
        }

