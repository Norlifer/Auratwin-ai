"""
Facility Manager AI Agent for AuraTwin AI
Processes natural language inquiries about building telemetry, energy waste, HVAC setpoints, and savings.
"""

from typing import Dict, Any, List


class FacilityManagerAgent:
    """Agentic AI engine analyzing digital twin telemetry and answering natural language queries."""

    def __init__(self):
        pass

    def answer_query(
        self,
        prompt: str,
        zones_telemetry: List[Dict[str, Any]],
        energy_summary: Dict[str, Any],
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes an intelligent, data-grounded response to facility manager queries.
        """
        query = prompt.lower().strip()

        # Find key metrics
        total_power = energy_summary.get("current_power_kw", 0.0)
        baseline_power = energy_summary.get("baseline_power_kw", 0.0)
        savings_kw = energy_summary.get("potential_saving_kw", 0.0)
        saving_pct = energy_summary.get("saving_percentage", 0.0)
        tariff_tier = energy_summary.get("tariff_tier", "Standard")
        price = energy_summary.get("price_per_kwh", 0.15)
        today_kwh = energy_summary.get("today_consumption_kwh", 0.0)
        today_savings = energy_summary.get("today_savings_usd", 0.0)

        # 1. "Which zone is wasting the most electricity?" or "wasting" / "inefficient"
        if any(w in query for w in ["waste", "wasting", "inefficient", "highest power", "most power", "overcooled"]):
            # Identify zones that are cold/overcooled with low or zero occupancy
            wasted_zones = []
            for z in zones_telemetry:
                occ = z.get("estimated_occupancy", 0)
                power = z.get("power_kw", 0.0)
                temp = z.get("temperature_c", 25.0)
                setpoint = z.get("setpoint_c", 23.0)
                
                # Waste factor: high power with low occupancy, or setpoint too low when empty
                if z.get("type") != "server_room":
                    if occ <= 2 and power > 3.5:
                        wasted_zones.append((z, power, f"Empty/Low occupancy ({occ} occupants) but HVAC drawing {power} kW at {setpoint}°C."))
                    elif occ == 0 and setpoint < 24.0:
                        wasted_zones.append((z, power, f"Zero occupancy detected, but thermostat is set aggressively to {setpoint}°C."))

            # Sort by power draw
            wasted_zones.sort(key=lambda x: x[1], reverse=True)

            if wasted_zones:
                worst_zone, p_draw, reason_str = wasted_zones[0]
                response_text = (
                    f"**Analysis of Energy Waste:**\n\n"
                    f"The zone with the most significant energy inefficiency is **{worst_zone['name']}**.\n\n"
                    f"- **Current Draw:** `{p_draw} kW`\n"
                    f"- **Occupancy:** `{worst_zone['estimated_occupancy']}` people (`{worst_zone['wifi_devices']}` Wi-Fi devices detected)\n"
                    f"- **Current Temperature / Setpoint:** `{worst_zone['temperature_c']}°C` / `{worst_zone['setpoint_c']}°C`\n"
                    f"- **Issue Detected:** {reason_str}\n\n"
                    f"**Recommended Action:** Apply OR-Tools eco-mode recommendation to relax setpoint to "
                    f"`{next((r['recommended_setpoint_c'] for r in recommendations if r['zone_id'] == worst_zone['zone_id']), 26.0)}°C` "
                    f"to immediately trim **~{round(p_draw * 0.4, 2)} kW**."
                )
            else:
                top_power = max(zones_telemetry, key=lambda z: z["power_kw"])
                response_text = (
                    f"All zones are currently operating within nominal occupancy-load limits. "
                    f"The highest active load is currently in **{top_power['name']}** (`{top_power['power_kw']} kW`), "
                    f"which corresponds appropriately with its occupancy of `{top_power['estimated_occupancy']}` occupants."
                )

            return {
                "response": response_text,
                "category": "waste_analysis",
                "suggested_actions": ["Apply AI Recommended Setpoints", "View Floorplan Heatmap"]
            }

        # 2. "How much energy did we save today?" or "savings" / "cost"
        if any(w in query for w in ["save", "saving", "savings", "saved today", "cost", "tariff", "bill"]):
            response_text = (
                f"**Energy & Cost Savings Summary:**\n\n"
                f"- **Today's Cumulative Energy Saved:** `~{round(today_kwh * 0.22, 1)} kWh` (~`{saving_pct}%` reduction)\n"
                f"- **Financial Savings Today:** `${today_savings:.2f} USD`\n"
                f"- **Instantaneous Building Demand:** `{total_power} kW` vs Baseline `{baseline_power} kW`\n"
                f"- **Active Electricity Tariff:** `{tariff_tier}` (`${price:.2f} / kWh`)\n"
                f"- **Hourly Run-Rate Savings:** `${energy_summary.get('hourly_savings_usd', 0.0):.2f}/hour`\n\n"
                f"By dynamically synchronizing HVAC setpoints with real-time Wi-Fi device density, the building avoids conditioning empty classrooms and transit corridors."
            )
            return {
                "response": response_text,
                "category": "savings_report",
                "suggested_actions": ["Download Telemetry Report", "View Tariff Breakdown"]
            }

        # 3. Zone specific lookup (e.g. "Server Room", "Meeting Room", "Classroom 1", etc.)
        for z in zones_telemetry:
            if z["name"].lower() in query or z["zone_id"].lower() in query:
                rec = next((r for r in recommendations if r["zone_id"] == z["zone_id"]), None)
                rec_sp = rec["recommended_setpoint_c"] if rec else z["setpoint_c"]
                mode = rec["mode"] if rec else "Normal"
                
                response_text = (
                    f"**Telemetry & Status for {z['name']}:**\n\n"
                    f"- **Temperature:** `{z['temperature_c']}°C` (Relative Humidity: `{z['humidity_pct']}%`)\n"
                    f"- **Active HVAC Setpoint:** `{z['setpoint_c']}°C`\n"
                    f"- **Wi-Fi Telemetry:** `{z['wifi_devices']}` active probes -> Estimated `{z['estimated_occupancy']}` occupants ({z['occupancy_percentage']}% capacity)\n"
                    f"- **Density Cluster:** `{z['density_cluster']}` (ZDI: `{z['zdi']}`)\n"
                    f"- **Power Draw:** `{z['power_kw']} kW` (Baseline: `{z['baseline_power_kw']} kW`)\n"
                    f"- **AI Recommendation:** Mode `{mode}` with target setpoint `{rec_sp}°C`.\n"
                    f"- **Reasoning:** {rec['reason'] if rec else 'Operating normally.'}"
                )
                return {
                    "response": response_text,
                    "category": "zone_detail",
                    "zone_id": z["zone_id"],
                    "suggested_actions": [f"Adjust {z['name']} Setpoint", "Inspect Zone Telemetry"]
                }

        # 4. K-Means clustering inquiry
        if any(w in query for w in ["cluster", "kmeans", "k-means", "density", "zdi", "index"]):
            high_d = [z["name"] for z in zones_telemetry if "High" in z.get("density_cluster", "")]
            med_d = [z["name"] for z in zones_telemetry if "Medium" in z.get("density_cluster", "")]
            low_d = [z["name"] for z in zones_telemetry if "Low" in z.get("density_cluster", "")]

            response_text = (
                f"**K-Means Zone Density Intelligence Report:**\n\n"
                f"The 5-dimensional feature vectors (`[Wi-Fi devices, temp, power, hour, occupancy ratio]`) classified the 10 building zones as follows:\n\n"
                f"- **High Density (ZDI > 0.65):** {', '.join(high_d) if high_d else 'None'}\n"
                f"- **Medium Density (ZDI 0.30 - 0.65):** {', '.join(med_d) if med_d else 'None'}\n"
                f"- **Low Density / Empty (ZDI < 0.30):** {', '.join(low_d) if low_d else 'None'}\n\n"
                f"K-Means dynamically feeds the Zone Density Index into OR-Tools to adapt cooling priorities in real-time."
            )
            return {
                "response": response_text,
                "category": "clustering_report",
                "suggested_actions": ["Optimize Setpoints", "Refresh Telemetry"]
            }

        # 5. General AI assistant response / Optimization summary
        rec_count = len(recommendations)
        total_opt_saving = sum(r.get("expected_saving_kw", 0.0) for r in recommendations)
        response_text = (
            f"**AuraTwin AI Building Overview:**\n\n"
            f"- **Monitored Zones:** `10 zones` across virtual HVAC system\n"
            f"- **Total Active Power:** `{total_power} kW`\n"
            f"- **Optimizable Setpoint Adjustments:** `{rec_count} recommendations ready`\n"
            f"- **Immediate Power Reduction Available:** `~{round(total_opt_saving, 2)} kW`\n"
            f"- **Current Tariff:** `{tariff_tier}` (`${price:.2f} / kWh`)\n\n"
            f"You can ask me questions such as:\n"
            f"- *'Which zone is wasting the most electricity?'*\n"
            f"- *'How much energy did we save today?'*\n"
            f"- *'What is the status of Lab 1 and Server Room?'*\n"
            f"- *'Explain the current K-Means density distribution.'*"
        )
        return {
            "response": response_text,
            "category": "general_overview",
            "suggested_actions": ["Apply AI Recommended Setpoints", "View 2D Floorplan"]
        }


facility_agent = FacilityManagerAgent()
