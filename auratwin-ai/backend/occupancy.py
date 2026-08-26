"""
Occupancy Estimation Engine for AuraTwin AI
Converts raw Wi-Fi probe/connected device observations into calibrated occupancy metrics.
"""

from typing import Dict, Any, Tuple


class OccupancyEngine:
    """Estimates real-time occupancy and capacity metrics from Wi-Fi telemetry."""

    def __init__(self, device_to_person_ratio: float = 1.05):
        """
        :param device_to_person_ratio: Average number of Wi-Fi active devices per person
               (typically ~1.0 to 1.1 in commercial/educational buildings).
        """
        self.ratio = device_to_person_ratio

    def estimate_occupancy(self, wifi_devices: int, capacity: int) -> Dict[str, Any]:
        """
        Estimate occupancy for a given zone.
        estimated_occupancy = round(detected_devices / ratio)
        occupancy_percentage = (estimated_occupancy / capacity) * 100
        """
        if wifi_devices <= 0:
            estimated_count = 0
        else:
            estimated_count = max(1, int(round(wifi_devices / self.ratio)))

        # Cap at 130% capacity to account for overflow/crowding
        estimated_count = min(estimated_count, int(capacity * 1.3))
        
        occupancy_percentage = round((estimated_count / max(1, capacity)) * 100, 1)

        # Classification
        if estimated_count == 0:
            category = "Empty"
        elif occupancy_percentage < 30.0:
            category = "Low"
        elif occupancy_percentage < 70.0:
            category = "Medium"
        else:
            category = "High"

        return {
            "wifi_devices": wifi_devices,
            "estimated_occupancy": estimated_count,
            "capacity": capacity,
            "occupancy_percentage": occupancy_percentage,
            "category": category,
            "is_occupied": estimated_count > 0,
        }


# Default singleton instance
occupancy_engine = OccupancyEngine()
