# AuraTwin AI 🌐🏢
### Intelligent Building Digital Twin & Virtual HVAC Optimizer

AuraTwin AI bridges commercial Wi-Fi sensor telemetry, K-Means occupancy density intelligence, constrained OR-Tools HVAC optimization, and virtual BACnet physics to eliminate energy waste in commercial buildings.

---

## 🏗️ Architecture & MVP Capabilities

```
Wi-Fi Telemetry & Sensor Simulator
             │
             ▼
      FastAPI Backend
      ┌──────┴────────────────────────┐
      ▼                               ▼
Occupancy Estimation Engine     Energy & Tariff Engine (ToU)
      │                               │
      ▼                               ▼
K-Means Clustering (ZDI)        OR-Tools HVAC Optimizer
      │                               │
      └──────────────┬────────────────┘
                     ▼
       Virtual BACnet / HVAC Simulation
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
Dark-Mode 2D Thermal Dashboard   Facility Manager AI Agent
```

### Key Modules:
1. **10-Zone Building Model (`data/zones.json`)**:
   - Classroom 1 & 2, Lab 1 & 2, Office 1 & 2, Meeting Room, Server Room (safety locked), Corridor, Auditorium.
2. **Telemetry Generator (`backend/data_generator.py`)**:
   - Realistic Wi-Fi device counts, thermal loads, humidity, and baseline power.
3. **Occupancy Engine (`backend/occupancy.py`)**:
   - Converts Wi-Fi probe density into calibrated headcount & capacity utilization.
4. **K-Means Density Clustering (`backend/clustering.py`)**:
   - Calculates continuous **Zone Density Index (ZDI)** & 3-stage density clustering.
5. **Time-of-Use Energy Engine (`backend/energy.py` & `data/tariffs.csv`)**:
   - Dynamic 24-hour tariff pricing, baseline tracking, and real-time kW savings.
6. **OR-Tools HVAC Optimizer (`backend/optimizer.py`)**:
   - Constrained mathematical solver minimizing $\text{Energy Cost} + \text{Comfort Penalty}$.
7. **Virtual BACnet Layer (`backend/bacnet.py`)**:
   - Emulates BACnet Analog Value / Input objects with realistic thermodynamic physics.
8. **AI Facility Manager (`backend/agent.py`)**:
   - Conversational AI agent answering questions like *"Which zone is wasting the most electricity?"* or *"How much energy did we save today?"*.
9. **Interactive 2D Floorplan Dashboard (`backend/main.py`)**:
   - Real-time SVG thermal-density heatmaps, live zone controls, and chart analytics.

---

## 🚀 Quick Start

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the AuraTwin Server
```bash
cd backend
python main.py
```
Or run with uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Open the Interactive Digital Twin
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📡 Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/telemetry` | Live Wi-Fi, HVAC, and thermal metrics across 10 zones |
| `POST` | `/telemetry` | Ingest external sensor telemetry |
| `GET` | `/zones` | Metadata, floorplan coordinates, and HVAC specs |
| `GET` | `/occupancy` | Device counts, occupancy ratios, and K-Means ZDI |
| `GET` | `/energy` | Building power draw, baseline, tariffs, and savings |
| `GET` | `/recommendations` | OR-Tools HVAC setpoint recommendations |
| `POST` | `/bacnet/override` | Write setpoint to virtual BACnet controller (AV:1) |
| `POST` | `/bacnet/apply-all-recommendations` | Batch apply all AI setpoints |
| `POST` | `/simulate/step` | Advance virtual clock and thermal simulation |
| `POST` | `/ai/chat` | Facility Manager AI agent conversation endpoint |

---

## 💡 Example AI Facility Manager Queries
- *"Which zone is wasting the most electricity?"*
- *"How much energy did we save today?"*
- *"What is the status of Server Room and Meeting Room?"*
- *"Explain the current K-Means density distribution."*
