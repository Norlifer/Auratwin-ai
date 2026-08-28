"""CCTV occupancy estimation for AuraTwin AI.

The input is the number of people detected in the latest CCTV snapshot.  The
engine keeps the capacity classification and percentage calculation in one
place for the rest of the application.
"""

from typing import Dict, Any, Tuple


class OccupancyEngine:
    """Converts a CCTV person count into zone occupancy metrics."""

    def estimate_occupancy(self, detected_people: int, capacity: int) -> Dict[str, Any]:
        """
        Estimate occupancy for a given zone.

        The detector already returns people, so no conversion is performed here.
        """
        estimated_count = max(0, int(detected_people))

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
            "detected_people": estimated_count,
            "estimated_occupancy": estimated_count,
            "capacity": capacity,
            "occupancy_percentage": occupancy_percentage,
            "category": category,
            "is_occupied": estimated_count > 0,
        }


# Default singleton instance
occupancy_engine = OccupancyEngine()
