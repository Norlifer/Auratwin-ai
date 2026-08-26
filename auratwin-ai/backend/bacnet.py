"""
Virtual BACnet / HVAC Digital Twin Layer for AuraTwin AI
Simulates BACnet IP objects (Analog Input, Analog Value, Binary Value) and physical thermal dynamics for 10 zones.
"""

import json
import os
import random
from typing import Dict, Any, List


class VirtualBACnetZone:
    """Represents a virtual BACnet HVAC controller for a single zone."""

    def __init__(self, zone_config: Dict[str, Any]):
        self.zone_id = zone_config["id"]
        self.name = zone_config["name"]
        self.type = zone_config.get("type", "standard")
        self.capacity = zone_config["capacity"]
        self.area_sqm = zone_config["area_sqm"]
        self.max_kw = zone_config.get("hvac_power_kw", 8.0)
        self.base_temp_c = zone_config.get("base_temp_c", 26.5)

        # BACnet Object representations
        self.bacnet_device_id = 1000 + int(self.zone_id.replace("zone_", ""))
        self.present_value_temp = zone_config.get("current_temp_c", 26.0)
        self.setpoint_c = zone_config.get("target_temp_c", 23.0)
        self.hvac_status = "ACTIVE"
        self.fan_speed = "AUTO"
        self.manual_override = False

        # Physical constants
        self.thermal_inertia = 0.15  # Rate of temp approach per simulation step
        self.outdoor_temp_c = 29.5
        self.humidity_pct = 55.0

    def set_setpoint(self, new_setpoint: float, is_manual: bool = False):
        """Write to BACnet Analog Value (AV:1 - Temperature Setpoint)."""
        self.setpoint_c = round(new_setpoint, 1)
        if is_manual:
            self.manual_override = True

    def release_override(self):
        """Releases manual override and returns control to autonomous AI optimizer."""
        self.manual_override = False

    def step_physics(self, estimated_occupants: int):
        """
        Simulates 1 time-step of thermodynamic progression:
        T(t+1) = T(t) + alpha * (Setpoint - T(t)) + beta * (Outdoor - T(t)) + gamma * (Occupants)
        """
        # HVAC cooling drive towards setpoint
        cooling_pull = (self.setpoint_c - self.present_value_temp) * self.thermal_inertia

        # Ambient heat infiltration from outdoor/structure
        heat_infiltration = (self.outdoor_temp_c - self.present_value_temp) * 0.03

        # Human heat load (~0.01°C increase per occupant per step)
        occupant_heat = (estimated_occupants / max(1, self.capacity)) * 0.18

        # Server heat load for server room
        server_heat = 0.25 if self.type == "server_room" else 0.0

        # Small random noise (air currents, sensor jitter)
        noise = random.uniform(-0.04, 0.04)

        # Update present value
        new_temp = self.present_value_temp + cooling_pull + heat_infiltration + occupant_heat + server_heat + noise
        self.present_value_temp = round(new_temp, 2)

        # Update humidity dynamically (cooling dries air slightly)
        target_humidity = 50.0 if self.present_value_temp < 23.0 else 58.0
        self.humidity_pct = round(self.humidity_pct + 0.1 * (target_humidity - self.humidity_pct) + random.uniform(-0.3, 0.3), 1)

    def get_bacnet_state(self) -> Dict[str, Any]:
        """Returns standard BACnet object model telemetry."""
        return {
            "bacnet_device_id": self.bacnet_device_id,
            "zone_id": self.zone_id,
            "zone_name": self.name,
            "objects": {
                "AI_1_temperature": round(self.present_value_temp, 2),
                "AI_2_humidity": round(self.humidity_pct, 1),
                "AV_1_setpoint": round(self.setpoint_c, 1),
                "BV_1_hvac_status": self.hvac_status,
                "BV_2_manual_override": self.manual_override,
            }
        }


class VirtualBACnetBuilding:
    """Manages the network of 10 virtual BACnet HVAC controllers."""

    def __init__(self, zones_config_path: str = "data/zones.json"):
        self.zones_config_path = zones_config_path
        self.controllers: Dict[str, VirtualBACnetZone] = {}
        self.load_zones()

    def load_zones(self):
        if os.path.exists(self.zones_config_path):
            with open(self.zones_config_path, "r", encoding="utf-8") as f:
                zones_data = json.load(f)
                for zd in zones_data:
                    self.controllers[zd["id"]] = VirtualBACnetZone(zd)

    def write_bacnet_setpoint(self, zone_id: str, setpoint: float, is_manual: bool = False) -> bool:
        """Emulates writing to BACnet controller over IP."""
        if zone_id in self.controllers:
            self.controllers[zone_id].set_setpoint(setpoint, is_manual=is_manual)
            return True
        return False

    def step_simulation(self, zone_occupancies: Dict[str, int]):
        """Steps all controllers forward in time."""
        for zid, ctrl in self.controllers.items():
            occ = zone_occupancies.get(zid, 0)
            ctrl.step_physics(occ)

    def get_all_states(self) -> List[Dict[str, Any]]:
        return [ctrl.get_bacnet_state() for ctrl in self.controllers.values()]


bacnet_building = VirtualBACnetBuilding()
