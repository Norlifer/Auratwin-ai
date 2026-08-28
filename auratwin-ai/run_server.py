"""
AuraTwin AI Launcher
Starts the AuraTwin AI FastAPI server and automatically locates your Python environment.
"""

import os
import sys
import subprocess
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")

# Keep CCTV automation progress visible alongside Uvicorn output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Ensure backend directory is in python path
sys.path.insert(0, backend_dir)
os.chdir(current_dir)

try:
    import uvicorn
    from backend.main import app
    print("================================================================")
    print("🚀 Starting AuraTwin AI Building Digital Twin & Virtual HVAC Engine")
    print("📡 Server running at: http://localhost:8000")
    print("📊 Interactive 2D Thermal Floorplan: http://localhost:8000/dashboard")
    print("================================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please install requirements using: pip install -r requirements.txt")
