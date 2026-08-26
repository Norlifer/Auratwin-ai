"""
Energy Model and Tariff Engine for AuraTwin AI
Calculates real-time power draw, baseline comparisons, potential savings, and cost metrics based on Time-of-Use tariffs.
"""

import csv
import os
from typing import Dict, Any, List


class EnergyEngine:
    """Computes energy demand, baseline consumption, cost, and efficiency savings."""

    def __init__(self, tariffs_path: str = "data/tariffs.csv"):
        self.tariffs_path = tariffs_path
        self.tariffs: Dict[int, Dict[str, Any]] = {}
        self.load_tariffs()

    def load_tariffs(self):
        """Loads hourly electricity tariffs from CSV."""
        if os.path.exists(self.tariffs_path):
            with open(self.tariffs_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hour = int(row["hour"])
                    self.tariffs[hour] = {
                        "hour": hour,
                        "time_window": row["time_window"],
                        "tier": row["tariff_tier"],
                        "price_per_kwh": float(row["price_per_kwh_usd"]),
                    }
        else:
            # Fallback standard tariffs
            for h in range(24):
                tier = "Peak" if 14 <= h < 20 else ("Standard" if 7 <= h < 23 else "Off-Peak")
                price = 0.28 if tier == "Peak" else (0.15 if tier == "Standard" else 0.08)
                self.tariffs[h] = {
                    "hour": h,
                    "time_window": f"{h:02d}:00 - {(h+1)%24:02d}:00",
                    "tier": tier,
                    "price_per_kwh": price,
                }

    def get_current_tariff(self, hour: int) -> Dict[str, Any]:
        """Returns the tariff info for a specific hour of the day."""
        h = hour % 24
        return self.tariffs.get(h, {
            "hour": h,
            "time_window": f"{h:02d}:00 - {(h+1)%24:02d}:00",
            "tier": "Standard",
            "price_per_kwh": 0.15,
        })

    def calculate_zone_power(
        self,
        zone_specs: Dict[str, Any],
        current_temp: float,
        setpoint: float,
        occupancy_ratio: float,
        hvac_on: bool = True
    ) -> float:
        """
        Calculates instantaneous power in kW for a single zone.
        Power = Standby + Thermal Workload (delta between current and setpoint) + Occupancy Internal Heat Gain
        """
        if not hvac_on:
            return round(0.1, 2)  # Minor standby / air sensor power

        max_kw = zone_specs.get("hvac_power_kw", 8.0)
        temp_delta = max(0.0, current_temp - setpoint)
        
        # Base HVAC operation power (30% base load when active)
        base_load = 0.30 * max_kw

        # Dynamic cooling effort needed to reach setpoint
        cooling_effort = min(0.55 * max_kw, (temp_delta / 4.0) * (0.55 * max_kw))

        # Additional load to remove human sensible heat (~100W per person equivalent)
        human_load = min(0.15 * max_kw, occupancy_ratio * (0.15 * max_kw))

        total_power = base_load + cooling_effort + human_load
        return round(min(max_kw, total_power), 2)

    def calculate_baseline_power(self, zone_specs: Dict[str, Any], current_temp: float) -> float:
        """
        Calculates baseline unoptimized power (traditional constant setpoint 21.5°C regardless of occupancy).
        """
        static_setpoint = 21.5
        max_kw = zone_specs.get("hvac_power_kw", 8.0)
        temp_delta = max(0.0, current_temp - static_setpoint)
        base_power = 0.40 * max_kw + (temp_delta / 4.0) * (0.60 * max_kw)
        return round(min(max_kw, base_power), 2)

    def calculate_building_energy_summary(
        self,
        zone_telemetry: List[Dict[str, Any]],
        current_hour: int
    ) -> Dict[str, Any]:
        """
        Aggregates total building power, baseline power, savings, and monetary impact.
        """
        tariff = self.get_current_tariff(current_hour)
        price_per_kwh = tariff["price_per_kwh"]

        current_total_kw = sum(z.get("power_kw", 0.0) for z in zone_telemetry)
        baseline_total_kw = sum(z.get("baseline_power_kw", z.get("power_kw", 0.0) * 1.35) for z in zone_telemetry)
        predicted_opt_kw = sum(z.get("optimized_power_kw", z.get("power_kw", 0.0) * 0.85) for z in zone_telemetry)

        potential_saving_kw = max(0.0, baseline_total_kw - current_total_kw)
        saving_percentage = round((potential_saving_kw / max(0.1, baseline_total_kw)) * 100, 1)

        hourly_cost = round(current_total_kw * price_per_kwh, 2)
        hourly_baseline_cost = round(baseline_total_kw * price_per_kwh, 2)
        hourly_savings_usd = round(max(0.0, hourly_baseline_cost - hourly_cost), 2)

        # Extrapolated daily stats
        daily_kwh = round(current_total_kw * 14.5, 1)  # average operating weighted factor
        daily_baseline_kwh = round(baseline_total_kw * 14.5, 1)
        daily_cost_usd = round(daily_kwh * price_per_kwh, 2)
        daily_saved_usd = round((daily_baseline_kwh - daily_kwh) * price_per_kwh, 2)

        return {
            "current_power_kw": round(current_total_kw, 2),
            "baseline_power_kw": round(baseline_total_kw, 2),
            "predicted_power_kw": round(predicted_opt_kw, 2),
            "potential_saving_kw": round(potential_saving_kw, 2),
            "saving_percentage": saving_percentage,
            "tariff_tier": tariff["tier"],
            "price_per_kwh": price_per_kwh,
            "time_window": tariff["time_window"],
            "hourly_cost_usd": hourly_cost,
            "hourly_savings_usd": hourly_savings_usd,
            "today_consumption_kwh": daily_kwh,
            "today_savings_usd": daily_saved_usd,
        }


energy_engine = EnergyEngine()
