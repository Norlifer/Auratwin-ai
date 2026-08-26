"""
Telemetry Data Simulator for AuraTwin AI
Generates realistic multi-zone Wi-Fi device telemetry, HVAC metrics, and energy consumption streams.
"""

import csv
from datetime import datetime, timedelta
import json
import os
import random
from typing import List, Dict, Any

from occupancy import occupancy_engine
from clustering import kmeans_engine
from energy import energy_engine
from bacnet import bacnet_building


class TelemetrySimulator:
    """Simulates real-time & historical Wi-Fi, HVAC, and thermal telemetry for 10 zones."""

    def __init__(self, zones_path: str = "data/zones.json", seed_csv_path: str = "data/telemetry.csv"):
        self.zones_path = zones_path
        self.seed_csv_path = seed_csv_path
        self.zones = self._load_zones()
        self.current_simulation_time = datetime.now().replace(hour=14, minute=15, second=0, microsecond=0)

    def _load_zones(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.zones_path):
            with open(self.zones_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _get_occupancy_profile(self, zone_type: str, hour: int, capacity: int) -> int:
        """Determines expected Wi-Fi device load based on zone type and time of day."""
        # Night time (22:00 - 06:00)
        if hour >= 22 or hour < 6:
            if zone_type == "server_room":
                return random.randint(0, 2)
            elif zone_type == "corridor":
                return random.randint(0, 3)
            return random.randint(0, 1)

        # Working hours (08:00 - 18:00)
        if 8 <= hour <= 18:
            if zone_type == "classroom":
                # High occupancy during classes
                ratio = random.uniform(0.50, 0.95) if hour in [9, 10, 11, 14, 15, 16] else random.uniform(0.1, 0.4)
            elif zone_type == "laboratory":
                ratio = random.uniform(0.40, 0.90) if 10 <= hour <= 17 else random.uniform(0.1, 0.3)
            elif zone_type == "office":
                ratio = random.uniform(0.60, 0.95) if 9 <= hour <= 17 else random.uniform(0.2, 0.4)
            elif zone_type == "meeting_room":
                ratio = random.uniform(0.50, 0.95) if hour in [10, 11, 14, 15, 16] else random.uniform(0.0, 0.25)
            elif zone_type == "server_room":
                return random.randint(1, 3)
            elif zone_type == "corridor":
                ratio = random.uniform(0.15, 0.50)
            elif zone_type == "auditorium":
                ratio = random.uniform(0.60, 0.95) if hour in [14, 15, 16] else random.uniform(0.0, 0.1)
            else:
                ratio = random.uniform(0.2, 0.6)

            devices = int(round(capacity * ratio * random.uniform(0.95, 1.15)))
            return max(0, min(int(capacity * 1.3), devices))

        # Early morning / late evening
        return max(0, int(capacity * random.uniform(0.05, 0.25)))

    def generate_current_telemetry(self) -> List[Dict[str, Any]]:
        """Generates a complete telemetry snapshot across all 10 zones."""
        hour = self.current_simulation_time.hour
        time_str = self.current_simulation_time.strftime("%H:%M:%S")
        timestamp_iso = self.current_simulation_time.isoformat()

        zone_telemetry_list = []
        occupancy_map = {}

        # 1. Generate Wi-Fi detections and occupancy for each zone
        for z in self.zones:
            zid = z["id"]
            wifi_devs = self._get_occupancy_profile(z.get("type", "classroom"), hour, z["capacity"])
            occ_info = occupancy_engine.estimate_occupancy(wifi_devs, z["capacity"])
            occupancy_map[zid] = occ_info["estimated_occupancy"]

            # 2. Get BACnet state
            ctrl = bacnet_building.controllers.get(zid)
            current_temp = ctrl.present_value_temp if ctrl else z.get("current_temp_c", 25.0)
            setpoint = ctrl.setpoint_c if ctrl else z.get("target_temp_c", 23.0)
            humidity = ctrl.humidity_pct if ctrl else 55.0

            # 3. Calculate Power Draw
            occ_ratio = occ_info["occupancy_percentage"] / 100.0
            power_kw = energy_engine.calculate_zone_power(z, current_temp, setpoint, occ_ratio)
            baseline_kw = energy_engine.calculate_baseline_power(z, current_temp)

            # 4. K-Means density prediction & ZDI
            features = [float(wifi_devs), float(current_temp), float(power_kw), float(hour), float(occ_ratio)]
            cluster_id, cluster_name, zdi = kmeans_engine.predict(features)

            zone_telemetry_list.append({
                "zone_id": zid,
                "name": z["name"],
                "type": z.get("type", "classroom"),
                "timestamp": timestamp_iso,
                "time": time_str,
                "wifi_devices": wifi_devs,
                "estimated_occupancy": occ_info["estimated_occupancy"],
                "capacity": z["capacity"],
                "occupancy_percentage": occ_info["occupancy_percentage"],
                "occupancy_category": occ_info["category"],
                "temperature_c": round(current_temp, 2),
                "setpoint_c": round(setpoint, 1),
                "humidity_pct": round(humidity, 1),
                "power_kw": power_kw,
                "baseline_power_kw": baseline_kw,
                "hvac_status": "ACTIVE" if power_kw > 1.0 else "STANDBY",
                "cluster_id": cluster_id,
                "density_cluster": cluster_name,
                "zdi": zdi,
                "area_sqm": z["area_sqm"],
                "floorplan": z["floorplan"],
                "manual_override": ctrl.manual_override if ctrl else False,
            })

        # Step thermal physics in virtual BACnet
        bacnet_building.step_simulation(occupancy_map)

        return zone_telemetry_list

    def step_simulation(self, minutes: int = 1) -> List[Dict[str, Any]]:
        """Advances virtual simulation clock and generates new telemetry."""
        self.current_simulation_time += timedelta(minutes=minutes)
        return self.generate_current_telemetry()

    def generate_seed_csv(self, days: int = 2):
        """Generates historical CSV data for testing and K-Means training."""
        os.makedirs(os.path.dirname(self.seed_csv_path), exist_ok=True)
        start_time = datetime.now() - timedelta(days=days)
        records = []
        training_features = []

        curr = start_time
        while curr <= datetime.now():
            hour = curr.hour
            time_str = curr.strftime("%Y-%m-%d %H:%M:%S")

            for z in self.zones:
                wifi_devs = self._get_occupancy_profile(z.get("type", "classroom"), hour, z["capacity"])
                occ_info = occupancy_engine.estimate_occupancy(wifi_devs, z["capacity"])
                occ_ratio = occ_info["occupancy_percentage"] / 100.0

                base_t = z.get("base_temp_c", 26.0)
                temp = round(base_t - random.uniform(0.5, 3.5) + (occ_ratio * 1.5), 2)
                setpoint = 23.0
                power_kw = energy_engine.calculate_zone_power(z, temp, setpoint, occ_ratio)

                records.append({
                    "timestamp": time_str,
                    "zone_id": z["id"],
                    "zone_name": z["name"],
                    "wifi_devices": wifi_devs,
                    "estimated_occupancy": occ_info["estimated_occupancy"],
                    "temperature_c": temp,
                    "humidity_pct": round(50.0 + random.uniform(0, 15), 1),
                    "power_kw": power_kw,
                    "hvac_status": "ACTIVE",
                    "setpoint_c": setpoint,
                })

                training_features.append([float(wifi_devs), float(temp), float(power_kw), float(hour), float(occ_ratio)])

            curr += timedelta(minutes=15)

        # Write to CSV
        if records:
            with open(self.seed_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        # Fit K-Means on seed features
        if training_features:
            kmeans_engine.fit(training_features)


telemetry_simulator = TelemetrySimulator()
# Generate seed historical data immediately
telemetry_simulator.generate_seed_csv(days=1)
