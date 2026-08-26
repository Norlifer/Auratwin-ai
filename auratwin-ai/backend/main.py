"""
AuraTwin AI - Core FastAPI Application Server
Provides REST APIs for Wi-Fi Telemetry, Occupancy Intelligence, OR-Tools Optimization, Virtual BACnet, and AI Facility Manager.
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from data_generator import telemetry_simulator
from occupancy import occupancy_engine
from clustering import kmeans_engine
from energy import energy_engine
from optimizer import hvac_optimizer
from bacnet import bacnet_building
from agent import facility_agent

app = FastAPI(
    title="AuraTwin AI API",
    description="Intelligent Digital Twin & Virtual HVAC Optimizer for Commercial Buildings",
    version="1.0.0"
)

# Enable CORS for Next.js / frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request Models
class TelemetryInput(BaseModel):
    zone: str
    wifi_devices: int
    temperature: float
    humidity: Optional[float] = 55.0
    power_kw: Optional[float] = None
    setpoint: Optional[float] = 23.0


class SetpointOverrideInput(BaseModel):
    zone_id: str
    setpoint_c: float
    is_manual: Optional[bool] = True


class ChatInput(BaseModel):
    message: str


class StepInput(BaseModel):
    minutes: Optional[int] = 1


# Helper to get current state
def get_current_system_state():
    telemetry = telemetry_simulator.generate_current_telemetry()
    hour = telemetry_simulator.current_simulation_time.hour
    energy_summary = energy_engine.calculate_building_energy_summary(telemetry, hour)
    is_peak = energy_summary["tariff_tier"] == "Peak"
    recommendations = hvac_optimizer.optimize_zone_setpoints(
        telemetry, energy_summary["price_per_kwh"], is_peak_tariff=is_peak
    )
    return telemetry, energy_summary, recommendations


# --- REST API Endpoints ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AuraTwin AI Digital Twin Engine",
        "zones_count": len(telemetry_simulator.zones),
        "simulation_time": telemetry_simulator.current_simulation_time.isoformat()
    }


@app.get("/zones")
def get_zones():
    """Retrieve metadata and physical specifications for all 10 building zones."""
    return telemetry_simulator.zones


@app.get("/zones/{zone_id}")
def get_zone_by_id(zone_id: str):
    """Retrieve details for a specific zone."""
    for z in telemetry_simulator.zones:
        if z["id"] == zone_id:
            ctrl = bacnet_building.controllers.get(zone_id)
            bacnet_data = ctrl.get_bacnet_state() if ctrl else {}
            return {"zone_spec": z, "bacnet_state": bacnet_data}
    raise HTTPException(status_code=404, detail="Zone not found")


@app.get("/telemetry")
def get_telemetry():
    """Retrieve the latest real-time Wi-Fi, HVAC, and thermal telemetry across all zones."""
    telemetry, energy_summary, _ = get_current_system_state()
    return {
        "simulation_time": telemetry_simulator.current_simulation_time.isoformat(),
        "time_str": telemetry_simulator.current_simulation_time.strftime("%H:%M:%S"),
        "telemetry": telemetry,
        "energy_summary": energy_summary
    }


@app.post("/telemetry")
def post_telemetry(payload: TelemetryInput):
    """Ingest real or simulated external Wi-Fi / temperature sensor telemetry."""
    zone_id = payload.zone if payload.zone.startswith("zone_") else f"zone_{payload.zone.lower().replace(' ', '_')}"
    ctrl = bacnet_building.controllers.get(zone_id)
    if not ctrl:
        # Try finding by name
        for z in telemetry_simulator.zones:
            if z["name"].lower() == payload.zone.lower():
                zone_id = z["id"]
                ctrl = bacnet_building.controllers.get(zone_id)
                break

    if ctrl:
        ctrl.present_value_temp = payload.temperature
        if payload.humidity:
            ctrl.humidity_pct = payload.humidity
        if payload.setpoint:
            ctrl.setpoint_c = payload.setpoint

    return {
        "status": "ingested",
        "zone_id": zone_id,
        "received": payload.dict()
    }


@app.get("/occupancy")
def get_occupancy():
    """Retrieve Wi-Fi based occupancy estimations, capacity ratios, and K-Means density indices."""
    telemetry, _, _ = get_current_system_state()
    total_devices = sum(z["wifi_devices"] for z in telemetry)
    total_occupancy = sum(z["estimated_occupancy"] for z in telemetry)
    total_capacity = sum(z["capacity"] for z in telemetry)
    
    return {
        "total_wifi_devices": total_devices,
        "total_estimated_occupancy": total_occupancy,
        "total_capacity": total_capacity,
        "building_occupancy_percentage": round((total_occupancy / max(1, total_capacity)) * 100, 1),
        "zones": [
            {
                "zone_id": z["zone_id"],
                "name": z["name"],
                "wifi_devices": z["wifi_devices"],
                "estimated_occupancy": z["estimated_occupancy"],
                "capacity": z["capacity"],
                "occupancy_percentage": z["occupancy_percentage"],
                "category": z["occupancy_category"],
                "density_cluster": z["density_cluster"],
                "zdi": z["zdi"]
            }
            for z in telemetry
        ]
    }


@app.get("/energy")
def get_energy():
    """Retrieve building energy consumption, baseline comparisons, ToU tariffs, and potential savings."""
    telemetry, energy_summary, _ = get_current_system_state()
    return {
        "summary": energy_summary,
        "zone_breakdown": [
            {
                "zone_id": z["zone_id"],
                "name": z["name"],
                "power_kw": z["power_kw"],
                "baseline_power_kw": z["baseline_power_kw"],
                "saving_kw": round(max(0.0, z["baseline_power_kw"] - z["power_kw"]), 2),
                "hvac_status": z["hvac_status"],
                "setpoint_c": z["setpoint_c"],
                "temperature_c": z["temperature_c"]
            }
            for z in telemetry
        ]
    }


@app.get("/recommendations")
def get_recommendations():
    """Retrieve OR-Tools optimized HVAC setpoints and eco setback strategies."""
    _, energy_summary, recommendations = get_current_system_state()
    return {
        "tariff_tier": energy_summary["tariff_tier"],
        "price_per_kwh": energy_summary["price_per_kwh"],
        "total_recommendations": len(recommendations),
        "total_potential_savings_kw": round(sum(r["expected_saving_kw"] for r in recommendations), 2),
        "recommendations": recommendations
    }


@app.post("/bacnet/override")
def override_bacnet_setpoint(payload: SetpointOverrideInput):
    """Write target setpoint to Virtual BACnet controller Analog Value object."""
    success = bacnet_building.write_bacnet_setpoint(
        payload.zone_id, payload.setpoint_c, is_manual=payload.is_manual or False
    )
    if not success:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {
        "status": "applied",
        "zone_id": payload.zone_id,
        "new_setpoint_c": payload.setpoint_c,
        "mode": "MANUAL_OVERRIDE" if payload.is_manual else "AI_OPTIMIZED"
    }


@app.post("/bacnet/apply-all-recommendations")
def apply_all_recommendations():
    """Batch applies all AI optimizer recommendations to the virtual BACnet controllers."""
    _, _, recommendations = get_current_system_state()
    applied = []
    for rec in recommendations:
        zid = rec["zone_id"]
        target_sp = rec["recommended_setpoint_c"]
        bacnet_building.write_bacnet_setpoint(zid, target_sp, is_manual=False)
        applied.append({"zone_id": zid, "setpoint_c": target_sp})
    return {
        "status": "all_applied",
        "count": len(applied),
        "applied": applied
    }


@app.post("/simulate/step")
def simulate_step(payload: StepInput = Body(default=StepInput(minutes=1))):
    """Advances virtual building simulation clock and triggers thermal physics progression."""
    telemetry = telemetry_simulator.step_simulation(minutes=payload.minutes or 1)
    hour = telemetry_simulator.current_simulation_time.hour
    energy_summary = energy_engine.calculate_building_energy_summary(telemetry, hour)
    return {
        "simulation_time": telemetry_simulator.current_simulation_time.isoformat(),
        "time_str": telemetry_simulator.current_simulation_time.strftime("%H:%M:%S"),
        "step_minutes": payload.minutes,
        "total_power_kw": energy_summary["current_power_kw"],
        "total_occupancy": sum(z["estimated_occupancy"] for z in telemetry)
    }


@app.post("/ai/chat")
def ai_chat(payload: ChatInput):
    """Conversational interface for Facility Managers asking natural language questions."""
    telemetry, energy_summary, recommendations = get_current_system_state()
    agent_output = facility_agent.answer_query(
        payload.message, telemetry, energy_summary, recommendations
    )
    return agent_output


# --- Interactive Dark Mode Dashboard UI ---
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serves an embedded, interactive dark-mode 2D Floorplan & Digital Twin Dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AuraTwin AI | Virtual HVAC & Building Digital Twin</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 50: '#ecfeff', 500: '#06b6d4', 600: '#0891b2', 900: '#164e63' },
                        dark: { 900: '#0B0F17', 800: '#111827', 700: '#1F2937', 600: '#374151' }
                    }
                }
            }
        }
    </script>
    <style>
        body { background-color: #0B0F17; color: #F3F4F6; font-family: system-ui, -apple-system, sans-serif; }
        .glass-card { background: rgba(17, 24, 39, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; }
        .zone-rect { transition: all 0.3s ease; cursor: pointer; }
        .zone-rect:hover { filter: brightness(1.25); stroke: #38bdf8; stroke-width: 2.5; }
        .pulse-dot { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.15); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #111827; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
    </style>
</head>
<body class="min-h-screen p-4 md:p-6 text-slate-100">

    <!-- Header Navigation -->
    <header class="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-gray-800">
        <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-xl bg-cyan-500/20 border border-cyan-400/40 flex items-center justify-center text-cyan-400 text-xl font-bold">
                <i class="fa-solid fa-cube"></i>
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <h1 class="text-xl font-bold tracking-tight text-white">AuraTwin AI</h1>
                    <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">10-Zone Digital Twin</span>
                    <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                        <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping"></span> Live Virtual BACnet
                    </span>
                </div>
                <p class="text-xs text-slate-400">Wi-Fi Telemetry &bull; K-Means Density &bull; OR-Tools Setpoint Optimization</p>
            </div>
        </div>

        <!-- Simulation Controls -->
        <div class="flex items-center gap-3">
            <div class="bg-gray-800/80 px-3 py-1.5 rounded-lg border border-gray-700 flex items-center gap-2 text-xs">
                <i class="fa-regular fa-clock text-cyan-400"></i>
                <span>Sim Clock: <strong id="simClock" class="text-cyan-300">--:--:--</strong></span>
            </div>
            <button onclick="advanceSimulation(1)" class="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs rounded-lg border border-gray-700 transition flex items-center gap-1.5">
                <i class="fa-solid fa-forward-step text-slate-300"></i> +1 Min
            </button>
            <button onclick="advanceSimulation(15)" class="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs rounded-lg border border-gray-700 transition flex items-center gap-1.5">
                <i class="fa-solid fa-forward text-slate-300"></i> +15 Min
            </button>
            <button onclick="toggleAutoTick()" id="autoTickBtn" class="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold rounded-lg shadow-lg shadow-cyan-500/20 transition flex items-center gap-1.5 text-white">
                <i class="fa-solid fa-play"></i> Auto Sim
            </button>
            <button onclick="applyAllRecommendations()" class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold rounded-lg shadow-lg shadow-emerald-500/20 transition flex items-center gap-1.5 text-white">
                <i class="fa-solid fa-bolt"></i> Auto-Apply AI Setpoints
            </button>
        </div>
    </header>

    <!-- Top Key Metrics Cards -->
    <div class="grid grid-cols-2 md:grid-cols-5 gap-4 my-6">
        <div class="glass-card p-4">
            <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Building Power</span>
                <i class="fa-solid fa-bolt text-amber-400"></i>
            </div>
            <div class="text-2xl font-bold text-white"><span id="metricPower">--</span> <span class="text-xs font-normal text-slate-400">kW</span></div>
            <div class="text-xs text-slate-400 mt-1 flex items-center gap-1">
                <span>Baseline: <span id="metricBaseline" class="text-slate-300">--</span> kW</span>
            </div>
        </div>

        <div class="glass-card p-4">
            <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Potential Savings</span>
                <i class="fa-solid fa-leaf text-emerald-400"></i>
            </div>
            <div class="text-2xl font-bold text-emerald-400"><span id="metricSavings">--</span> <span class="text-xs font-normal text-slate-400">kW</span></div>
            <div class="text-xs text-emerald-500/90 mt-1 flex items-center gap-1">
                <i class="fa-solid fa-arrow-down text-[10px]"></i> <span id="metricSavingPct">--</span>% load reduction
            </div>
        </div>

        <div class="glass-card p-4">
            <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Active Tariff (ToU)</span>
                <i class="fa-solid fa-chart-line text-cyan-400"></i>
            </div>
            <div class="text-2xl font-bold text-cyan-300"><span id="metricTariff">--</span></div>
            <div class="text-xs text-slate-400 mt-1">
                $<span id="metricPrice">--</span> / kWh
            </div>
        </div>

        <div class="glass-card p-4">
            <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Est. Occupancy</span>
                <i class="fa-solid fa-users text-purple-400"></i>
            </div>
            <div class="text-2xl font-bold text-purple-300"><span id="metricOccupancy">--</span> <span class="text-xs font-normal text-slate-400">people</span></div>
            <div class="text-xs text-slate-400 mt-1">
                <span id="metricWifiDevices">--</span> Wi-Fi devices detected
            </div>
        </div>

        <div class="glass-card p-4">
            <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Today's Energy Saved</span>
                <i class="fa-solid fa-piggy-bank text-yellow-400"></i>
            </div>
            <div class="text-2xl font-bold text-yellow-300">$<span id="metricSavedDollars">--</span></div>
            <div class="text-xs text-slate-400 mt-1">
                <span id="metricTodayKwh">--</span> kWh total
            </div>
        </div>
    </div>

    <!-- Main Content Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Left: 2D Floorplan Thermal Visualizer (7 cols) -->
        <div class="lg:col-span-7 space-y-6">
            <div class="glass-card p-5">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-base font-semibold text-white flex items-center gap-2">
                            <i class="fa-solid fa-map text-cyan-400"></i> 2D Floorplan Thermal & Density Visualizer
                        </h2>
                        <p class="text-xs text-slate-400">Click any zone on blueprint to inspect telemetry or override setpoint</p>
                    </div>
                    <div class="flex items-center gap-2 text-xs">
                        <span class="flex items-center gap-1 text-slate-300"><span class="w-3 h-3 rounded bg-blue-500/50 inline-block"></span> Cool (&le;22°C)</span>
                        <span class="flex items-center gap-1 text-slate-300"><span class="w-3 h-3 rounded bg-emerald-500/50 inline-block"></span> Optimal</span>
                        <span class="flex items-center gap-1 text-slate-300"><span class="w-3 h-3 rounded bg-amber-500/50 inline-block"></span> Warm (&ge;26°C)</span>
                    </div>
                </div>

                <!-- SVG Interactive Floorplan Container -->
                <div class="relative w-full aspect-[920/520] bg-gray-950/80 rounded-xl border border-gray-800 p-2 overflow-hidden shadow-inner flex items-center justify-center">
                    <svg id="floorplanSvg" viewBox="0 0 920 520" class="w-full h-full">
                        <!-- Background Grid lines -->
                        <defs>
                            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255, 255, 255, 0.03)" stroke-width="1"/>
                            </pattern>
                        </defs>
                        <rect width="920" height="520" fill="url(#grid)" />
                        <!-- Zones rendered dynamically -->
                        <g id="zonesSvgGroup"></g>
                    </svg>
                </div>
            </div>

            <!-- Energy & Optimization Real-Time Chart -->
            <div class="glass-card p-5">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-semibold text-white flex items-center gap-2">
                        <i class="fa-solid fa-chart-area text-cyan-400"></i> Real-Time Demand vs AI Optimized Baseline (kW)
                    </h3>
                    <span class="text-xs text-slate-400">Live Telemetry Stream</span>
                </div>
                <div class="h-56 w-full">
                    <canvas id="energyChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Right: AI Recommendations, Selected Zone Modal & AI Chat (5 cols) -->
        <div class="lg:col-span-5 space-y-6">
            
            <!-- AI Recommendation Action Center -->
            <div class="glass-card p-5">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-semibold text-white flex items-center gap-2">
                        <i class="fa-solid fa-microchip text-emerald-400"></i> OR-Tools Optimization Engine
                    </h3>
                    <span id="recBadge" class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono">10 Active</span>
                </div>
                <div id="recommendationsList" class="space-y-2.5 max-h-64 overflow-y-auto pr-1 text-xs">
                    <!-- Loaded dynamically -->
                </div>
            </div>

            <!-- Selected Zone Inspection & BACnet Override Panel -->
            <div id="zoneDetailPanel" class="glass-card p-5 border-cyan-500/30">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-semibold text-white flex items-center gap-2">
                        <i class="fa-solid fa-sliders text-cyan-400"></i> <span id="detailZoneName">Classroom 1</span>
                    </h3>
                    <span id="detailZoneCluster" class="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">Medium Density</span>
                </div>

                <div class="grid grid-cols-2 gap-3 text-xs mb-4">
                    <div class="bg-gray-800/60 p-2.5 rounded-lg border border-gray-700/50">
                        <span class="text-slate-400">Temperature</span>
                        <div class="text-lg font-bold text-white mt-0.5"><span id="detailTemp">--</span>°C</div>
                        <span class="text-[10px] text-slate-400">Humidity: <span id="detailHumidity">--</span>%</span>
                    </div>
                    <div class="bg-gray-800/60 p-2.5 rounded-lg border border-gray-700/50">
                        <span class="text-slate-400">Wi-Fi & Occupancy</span>
                        <div class="text-lg font-bold text-purple-300 mt-0.5"><span id="detailOcc">--</span> / <span id="detailCap">--</span></div>
                        <span class="text-[10px] text-slate-400"><span id="detailWifi">--</span> active devices</span>
                    </div>
                </div>

                <!-- Setpoint Slider & BACnet Write -->
                <div class="space-y-2">
                    <div class="flex justify-between items-center text-xs">
                        <label class="text-slate-300">BACnet Setpoint (AV:1): <strong id="sliderVal" class="text-cyan-300">23.0</strong>°C</label>
                        <span id="detailOverrideBadge" class="text-[10px] text-slate-400">Auto AI</span>
                    </div>
                    <input type="range" id="setpointSlider" min="18.0" max="28.0" step="0.5" value="23.0" 
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                           oninput="document.getElementById('sliderVal').textContent = this.value">
                    <div class="flex gap-2 pt-1">
                        <button onclick="overrideSelectedSetpoint()" class="flex-1 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold rounded-lg text-white transition">
                            <i class="fa-solid fa-paper-plane mr-1"></i> Send BACnet AV:1 Write
                        </button>
                    </div>
                </div>
            </div>

            <!-- AI Facility Manager Conversational Assistant -->
            <div class="glass-card p-5 flex flex-col h-80">
                <div class="flex items-center justify-between mb-3 border-b border-gray-800 pb-2">
                    <div class="flex items-center gap-2">
                        <div class="h-6 w-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs">
                            <i class="fa-solid fa-robot"></i>
                        </div>
                        <h3 class="text-sm font-semibold text-white">Facility Manager AI Agent</h3>
                    </div>
                    <span class="text-[11px] text-emerald-400 flex items-center gap-1">
                        <span class="h-1.5 w-1.5 rounded-full bg-emerald-400"></span> Online
                    </span>
                </div>

                <!-- Chat history -->
                <div id="chatHistory" class="flex-1 overflow-y-auto space-y-3 pr-1 text-xs text-slate-200">
                    <div class="p-2.5 rounded-lg bg-gray-800/80 border border-gray-700 text-slate-300">
                        Hello! I am your <strong>AuraTwin AI Facility Manager</strong>. I monitor all 10 virtual HVAC zones, real-time Wi-Fi device density, and electricity tariffs. How can I assist you?
                    </div>
                </div>

                <!-- Quick Prompt Chips -->
                <div class="flex gap-1.5 py-2 overflow-x-auto text-[10px]">
                    <button onclick="sendPreset('Which zone is wasting the most electricity?')" class="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-slate-300 rounded border border-gray-700 whitespace-nowrap">
                        ⚡ Which zone is wasting power?
                    </button>
                    <button onclick="sendPreset('How much energy did we save today?')" class="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-slate-300 rounded border border-gray-700 whitespace-nowrap">
                        💰 Today's savings?
                    </button>
                    <button onclick="sendPreset('Explain K-Means cluster status')" class="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-slate-300 rounded border border-gray-700 whitespace-nowrap">
                        📊 K-Means clusters?
                    </button>
                </div>

                <!-- Chat Input -->
                <form onsubmit="handleChatSubmit(event)" class="flex gap-2 pt-1">
                    <input type="text" id="chatInput" placeholder="Ask AI about energy, zones, setpoints..." 
                           class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500">
                    <button type="submit" class="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold">
                        <i class="fa-solid fa-arrow-up"></i>
                    </button>
                </form>
            </div>

        </div>
    </div>

    <!-- JavaScript Client Logic -->
    <script>
        let currentTelemetry = [];
        let currentRecs = [];
        let selectedZoneId = "zone_1";
        let autoTickInterval = null;
        let chartInstance = null;
        let chartLabels = [];
        let chartPowerData = [];
        let chartBaselineData = [];

        // Color thermal scale for floorplan
        function getThermalColor(temp) {
            if (temp <= 21.5) return "rgba(59, 130, 246, 0.45)"; // Blue cool
            if (temp <= 23.5) return "rgba(16, 185, 129, 0.45)"; // Emerald optimal
            if (temp <= 25.5) return "rgba(245, 158, 11, 0.45)"; // Amber warm
            return "rgba(239, 68, 68, 0.55)"; // Red hot
        }

        function getThermalStroke(temp) {
            if (temp <= 21.5) return "#60a5fa";
            if (temp <= 23.5) return "#34d399";
            if (temp <= 25.5) return "#fbbf24";
            return "#f87171";
        }

        async function fetchState() {
            try {
                const res = await fetch('/telemetry');
                const data = await res.json();
                currentTelemetry = data.telemetry;
                document.getElementById('simClock').textContent = data.time_str;
                
                // Metrics
                const es = data.energy_summary;
                document.getElementById('metricPower').textContent = es.current_power_kw;
                document.getElementById('metricBaseline').textContent = es.baseline_power_kw;
                document.getElementById('metricSavings').textContent = es.potential_saving_kw;
                document.getElementById('metricSavingPct').textContent = es.saving_percentage;
                document.getElementById('metricTariff').textContent = es.tariff_tier;
                document.getElementById('metricPrice').textContent = es.price_per_kwh.toFixed(2);
                document.getElementById('metricTodayKwh').textContent = es.today_consumption_kwh;
                document.getElementById('metricSavedDollars').textContent = es.today_savings_usd.toFixed(2);

                const totalOcc = currentTelemetry.reduce((acc, z) => acc + z.estimated_occupancy, 0);
                const totalDev = currentTelemetry.reduce((acc, z) => acc + z.wifi_devices, 0);
                document.getElementById('metricOccupancy').textContent = totalOcc;
                document.getElementById('metricWifiDevices').textContent = totalDev;

                // Update Floorplan
                renderFloorplan();

                // Update Selected Zone Details
                updateZoneDetailPanel();

                // Fetch recommendations
                fetchRecommendations();

                // Update Chart
                updateChart(data.time_str, es.current_power_kw, es.baseline_power_kw);

            } catch (err) {
                console.error("Fetch state error:", err);
            }
        }

        async function fetchRecommendations() {
            try {
                const res = await fetch('/recommendations');
                const data = await res.json();
                currentRecs = data.recommendations;
                
                const list = document.getElementById('recommendationsList');
                list.innerHTML = '';

                data.recommendations.forEach(r => {
                    const item = document.createElement('div');
                    item.className = "p-2 rounded-lg bg-gray-900/90 border border-gray-800 flex items-center justify-between gap-2 hover:border-gray-700 transition";
                    item.innerHTML = `
                        <div class="flex-1">
                            <div class="flex items-center gap-1.5">
                                <span class="font-semibold text-slate-200">${r.zone_name}</span>
                                <span class="text-[10px] px-1.5 py-0.2 rounded bg-gray-800 text-slate-400">${r.mode}</span>
                            </div>
                            <div class="text-[11px] text-slate-400 mt-0.5">
                                Curr: <span class="text-slate-300">${r.current_setpoint_c}°C</span> &rarr; Rec: <strong class="text-emerald-400">${r.recommended_setpoint_c}°C</strong>
                                (${r.expected_saving_kw > 0 ? '+' + r.expected_saving_kw + ' kW saved' : 'Optimal'})
                            </div>
                        </div>
                        <button onclick="applySingleRecommendation('${r.zone_id}', ${r.recommended_setpoint_c})" class="px-2 py-1 bg-emerald-600/80 hover:bg-emerald-600 text-white rounded text-[11px] font-medium transition">
                            Apply
                        </button>
                    `;
                    list.appendChild(item);
                });
            } catch (e) {
                console.error("Recs error:", e);
            }
        }

        function renderFloorplan() {
            const group = document.getElementById('zonesSvgGroup');
            group.innerHTML = '';

            currentTelemetry.forEach(z => {
                const fp = z.floorplan;
                const fillColor = getThermalColor(z.temperature_c);
                const strokeColor = getThermalStroke(z.temperature_c);
                const isSelected = z.zone_id === selectedZoneId;

                const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
                g.setAttribute("class", "zone-rect");
                g.onclick = () => selectZone(z.zone_id);

                // Zone Box
                const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                rect.setAttribute("x", fp.x);
                rect.setAttribute("y", fp.y);
                rect.setAttribute("width", fp.width);
                rect.setAttribute("height", fp.height);
                rect.setAttribute("rx", "8");
                rect.setAttribute("fill", fillColor);
                rect.setAttribute("stroke", isSelected ? "#38bdf8" : strokeColor);
                rect.setAttribute("stroke-width", isSelected ? "3" : "1.5");
                g.appendChild(rect);

                // Zone Name Label
                const textName = document.createElementNS("http://www.w3.org/2000/svg", "text");
                textName.setAttribute("x", fp.x + 12);
                textName.setAttribute("y", fp.y + 24);
                textName.setAttribute("fill", "#FFFFFF");
                textName.setAttribute("font-size", "12");
                textName.setAttribute("font-weight", "bold");
                textName.textContent = z.name;
                g.appendChild(textName);

                // Temp & Setpoint
                const textTemp = document.createElementNS("http://www.w3.org/2000/svg", "text");
                textTemp.setAttribute("x", fp.x + 12);
                textTemp.setAttribute("y", fp.y + 44);
                textTemp.setAttribute("fill", "#E5E7EB");
                textTemp.setAttribute("font-size", "11");
                textTemp.textContent = `Temp: ${z.temperature_c}°C (SP: ${z.setpoint_c}°C)`;
                g.appendChild(textTemp);

                // Occupancy Badge & Wi-Fi
                const textOcc = document.createElementNS("http://www.w3.org/2000/svg", "text");
                textOcc.setAttribute("x", fp.x + 12);
                textOcc.setAttribute("y", fp.y + 62);
                textOcc.setAttribute("fill", "#C084FC");
                textOcc.setAttribute("font-size", "11");
                textOcc.textContent = `👥 ${z.estimated_occupancy}/${z.capacity} (${z.wifi_devices} Wi-Fi)`;
                g.appendChild(textOcc);

                // Power in kW
                const textPower = document.createElementNS("http://www.w3.org/2000/svg", "text");
                textPower.setAttribute("x", fp.x + 12);
                textPower.setAttribute("y", fp.y + 80);
                textPower.setAttribute("fill", "#FBBF24");
                textPower.setAttribute("font-size", "11");
                textPower.textContent = `⚡ ${z.power_kw} kW • ${z.density_cluster}`;
                g.appendChild(textPower);

                // Pulse dot if high density
                if (z.density_cluster === "High Density") {
                    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle.setAttribute("cx", fp.x + fp.width - 16);
                    circle.setAttribute("cy", fp.y + 16);
                    circle.setAttribute("r", "5");
                    circle.setAttribute("fill", "#f87171");
                    circle.setAttribute("class", "pulse-dot");
                    g.appendChild(circle);
                }

                group.appendChild(g);
            });
        }

        function selectZone(zoneId) {
            selectedZoneId = zoneId;
            renderFloorplan();
            updateZoneDetailPanel();
        }

        function updateZoneDetailPanel() {
            const z = currentTelemetry.find(item => item.zone_id === selectedZoneId);
            if (!z) return;

            document.getElementById('detailZoneName').textContent = z.name;
            document.getElementById('detailZoneCluster').textContent = `${z.density_cluster} (ZDI ${z.zdi})`;
            document.getElementById('detailTemp').textContent = z.temperature_c;
            document.getElementById('detailHumidity').textContent = z.humidity_pct;
            document.getElementById('detailOcc').textContent = z.estimated_occupancy;
            document.getElementById('detailCap').textContent = z.capacity;
            document.getElementById('detailWifi').textContent = z.wifi_devices;
            document.getElementById('setpointSlider').value = z.setpoint_c;
            document.getElementById('sliderVal').textContent = z.setpoint_c;
            document.getElementById('detailOverrideBadge').textContent = z.manual_override ? "Manual Override" : "Autonomous AI";
        }

        async function overrideSelectedSetpoint() {
            const val = parseFloat(document.getElementById('setpointSlider').value);
            try {
                await fetch('/bacnet/override', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ zone_id: selectedZoneId, setpoint_c: val, is_manual: true })
                });
                await fetchState();
            } catch (e) {
                console.error(e);
            }
        }

        async function applySingleRecommendation(zoneId, setpoint) {
            try {
                await fetch('/bacnet/override', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ zone_id: zoneId, setpoint_c: setpoint, is_manual: false })
                });
                await fetchState();
            } catch (e) {
                console.error(e);
            }
        }

        async function applyAllRecommendations() {
            try {
                await fetch('/bacnet/apply-all-recommendations', { method: 'POST' });
                await fetchState();
            } catch (e) {
                console.error(e);
            }
        }

        async function advanceSimulation(minutes) {
            try {
                await fetch('/simulate/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ minutes: minutes })
                });
                await fetchState();
            } catch (e) {
                console.error(e);
            }
        }

        function toggleAutoTick() {
            const btn = document.getElementById('autoTickBtn');
            if (autoTickInterval) {
                clearInterval(autoTickInterval);
                autoTickInterval = null;
                btn.innerHTML = '<i class="fa-solid fa-play"></i> Auto Sim';
                btn.className = "px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold rounded-lg text-white transition";
            } else {
                autoTickInterval = setInterval(() => advanceSimulation(2), 2500);
                btn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause Sim';
                btn.className = "px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-xs font-semibold rounded-lg text-white transition";
            }
        }

        function initChart() {
            const ctx = document.getElementById('energyChart').getContext('2d');
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartLabels,
                    datasets: [
                        {
                            label: 'Optimized Real-Time Power (kW)',
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.15)',
                            fill: true,
                            tension: 0.3,
                            data: chartPowerData,
                            borderWidth: 2
                        },
                        {
                            label: 'Unoptimized Baseline (kW)',
                            borderColor: '#ef4444',
                            borderDash: [5, 5],
                            data: chartBaselineData,
                            tension: 0.3,
                            fill: false,
                            borderWidth: 1.5
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af', font: { size: 10 } } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af', font: { size: 10 } }, min: 0 }
                    },
                    plugins: {
                        legend: { labels: { color: '#e5e7eb', font: { size: 11 } } }
                    }
                }
            });
        }

        function updateChart(timeStr, power, baseline) {
            if (chartLabels.length > 15) {
                chartLabels.shift();
                chartPowerData.shift();
                chartBaselineData.shift();
            }
            chartLabels.push(timeStr);
            chartPowerData.push(power);
            chartBaselineData.push(baseline);
            if (chartInstance) chartInstance.update();
        }

        async function handleChatSubmit(e) {
            e.preventDefault();
            const input = document.getElementById('chatInput');
            const msg = input.value.trim();
            if (!msg) return;
            input.value = '';

            appendChatMessage('user', msg);

            try {
                const res = await fetch('/ai/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                appendChatMessage('ai', data.response);
            } catch (err) {
                appendChatMessage('ai', 'Error contacting digital twin agent.');
            }
        }

        function sendPreset(text) {
            document.getElementById('chatInput').value = text;
            handleChatSubmit(new Event('submit'));
        }

        function appendChatMessage(sender, text) {
            const container = document.getElementById('chatHistory');
            const div = document.createElement('div');
            if (sender === 'user') {
                div.className = "p-2.5 rounded-lg bg-cyan-950/60 border border-cyan-800/60 text-cyan-200 ml-6 text-right";
                div.textContent = text;
            } else {
                div.className = "p-2.5 rounded-lg bg-gray-800/80 border border-gray-700 text-slate-200 mr-4";
                div.innerHTML = text.replace(/\\n/g, '<br/>').replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code class="bg-gray-900 px-1 py-0.5 rounded text-cyan-300">$1</code>');
            }
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        // Initial setup on load
        window.addEventListener('DOMContentLoaded', () => {
            initChart();
            fetchState();
            setInterval(fetchState, 3000);
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
