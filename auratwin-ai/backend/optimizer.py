"""Discrete mathematical HVAC optimizer for AuraTwin AI.

The optimizer evaluates constrained setpoint candidates and minimizes an
energy-cost plus comfort-penalty score.
"""

from typing import List, Dict, Any


class HVACOptimizer:
    """
    Optimizes HVAC setpoints across commercial zones.
    Objective function: Minimize Sum(Energy_Cost(setpoint_i, tariff) + Comfort_Penalty(setpoint_i, occupancy_i))
    Subject to:
        - min_setpoint_i <= setpoint_i <= max_setpoint_i
        - Special constraints (e.g., Server Room max temperature constraint <= 21.0°C)
    """

    def __init__(self):
        pass

    def optimize_zone_setpoints(
        self,
        zones_data: List[Dict[str, Any]],
        electricity_price: float,
        is_peak_tariff: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Calculates optimal setpoint recommendations for all zones.
        """
        recommendations = []

        for zone in zones_data:
            zone_id = zone.get("id")
            zone_name = zone.get("name")
            zone_type = zone.get("type", "standard")
            capacity = zone.get("capacity", 20)
            est_occ = zone.get("estimated_occupancy", 0)
            occ_ratio = zone.get("occupancy_percentage", 0.0) / 100.0
            current_temp = zone.get("current_temp_c", 25.0)
            current_setpoint = zone.get("setpoint_c", 23.0)
            min_sp = zone.get("min_setpoint_c", 20.0)
            max_sp = zone.get("max_setpoint_c", 27.0)
            comfort_weight = zone.get("comfort_weight", 0.8)
            zdi = zone.get("zdi", 0.3)

            # Special Zone Type Handling
            if zone_type == "server_room":
                # Critical thermal constraints for servers regardless of human presence
                rec_setpoint = 20.0
                reason = "Server room thermal safety lock: Maintain continuous cooling <= 20.5°C."
                mode = "Critical Cooling"
                expected_saving_kw = 0.0

            elif zone_type == "corridor":
                # Transitional space: higher eco flexibility
                rec_setpoint = 26.5 if is_peak_tariff else 25.5
                reason = "Corridor transit area: Eco setpoint active with minimal comfort impact."
                mode = "Eco Transit"
                expected_saving_kw = round(zone.get("hvac_power_kw", 4.0) * 0.40, 2)

            else:
                # Occupied commercial space optimization: grid search / discrete LP solver
                # Candidates every 0.5 deg C between min_sp and max_sp
                best_sp = current_setpoint
                min_total_cost = float("inf")
                base_target = zone.get("target_temp_c", 23.0)

                # Cost scaling factors
                energy_weight = 1.0 + (1.2 if is_peak_tariff else 0.4)

                step_count = int(round((max_sp - min_sp) / 0.5)) + 1
                for step in range(step_count):
                    candidate_sp = min_sp + (step * 0.5)

                    # 1. Energy Cost Component (lower setpoint = higher cooling energy)
                    # Baseline delta from ambient outdoor / unconditioned base temp (28°C)
                    cooling_effort = max(0.0, 28.0 - candidate_sp)
                    energy_cost = cooling_effort * energy_weight * electricity_price

                    # 2. Comfort Penalty Component
                    # Comfort dissatisfaction increases as setpoint deviates from ideal 22.5 - 23.5°C
                    temp_deviation = abs(candidate_sp - base_target)
                    
                    # If empty, comfort penalty is negligible (0.05)
                    # If high occupancy, comfort penalty is heavily weighted
                    effective_occ_factor = max(0.05, occ_ratio)
                    comfort_penalty = comfort_weight * effective_occ_factor * (temp_deviation ** 1.8) * 2.5

                    # If peak tariff and low occupancy, push towards eco setback
                    if is_peak_tariff and occ_ratio < 0.2:
                        energy_cost *= 1.3

                    total_cost = energy_cost + comfort_penalty

                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_sp = candidate_sp

                rec_setpoint = round(best_sp, 1)

                # Formulate intelligent justification
                if est_occ == 0:
                    mode = "Eco Standby"
                    reason = f"Zero occupancy detected via CCTV snapshot. Relaxed setpoint to {rec_setpoint}°C to eliminate phantom cooling."
                elif occ_ratio < 0.35:
                    mode = "Low-Density Dynamic"
                    reason = f"Low occupancy ({est_occ}/{capacity}). Setpoint optimized to {rec_setpoint}°C to preserve comfort while shaving peak load."
                elif occ_ratio > 0.75:
                    mode = "High Comfort Priority"
                    reason = f"High density ({est_occ}/{capacity}, ZDI {zdi}). Prioritizing occupant thermal comfort at {rec_setpoint}°C."
                else:
                    mode = "Balanced Standard"
                    reason = f"Moderate occupancy ({est_occ}/{capacity}). Setpoint balanced at {rec_setpoint}°C."

                delta_sp = rec_setpoint - current_setpoint
                expected_saving_kw = round(max(0.0, delta_sp * 0.65 * (zone.get("hvac_power_kw", 8.0) / 10.0)), 2)

            recommendations.append({
                "zone_id": zone_id,
                "zone_name": zone_name,
                "current_setpoint_c": current_setpoint,
                "recommended_setpoint_c": rec_setpoint,
                "current_temp_c": current_temp,
                "estimated_occupancy": est_occ,
                "occupancy_percentage": round(occ_ratio * 100, 1),
                "mode": mode,
                "reason": reason,
                "expected_saving_kw": expected_saving_kw,
                "tariff_adjusted": is_peak_tariff,
            })

        return recommendations


hvac_optimizer = HVACOptimizer()
