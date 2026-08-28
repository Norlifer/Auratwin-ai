const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ZoneTelemetry {
  zone_id: string;
  name: string;
  type: string;
  detected_people: number;
  snapshot_uploaded: boolean;
  snapshot_timestamp: string | null;
  estimated_occupancy: number;
  capacity: number;
  occupancy_percentage: number;
  occupancy_category: string;
  temperature_c: number;
  setpoint_c: number;
  humidity_pct: number;
  power_kw: number;
  baseline_power_kw: number;
  hvac_status: string;
  density_cluster: string;
  zdi: number;
  floorplan: { x: number; y: number; width: number; height: number };
  manual_override: boolean;
}

export interface EnergySummary {
  current_power_kw: number;
  baseline_power_kw: number;
  predicted_power_kw: number;
  potential_saving_kw: number;
  saving_percentage: number;
  tariff_tier: string;
  price_per_kwh: number;
  time_window: string;
  hourly_cost_usd: number;
  hourly_savings_usd: number;
  today_consumption_kwh: number;
  today_savings_usd: number;
}

export interface AIRecommendation {
  zone_id: string;
  zone_name: string;
  current_setpoint_c: number;
  recommended_setpoint_c: number;
  current_temp_c: number;
  estimated_occupancy: number;
  occupancy_percentage: number;
  mode: string;
  reason: string;
  expected_saving_kw: number;
  tariff_adjusted: boolean;
}

export interface SnapshotResult {
  status: string;
  zone_id: string;
  filename: string;
  people_count: number;
  head_count: number;
  width: number;
  height: number;
  timestamp: string;
  annotated_image_url: string;
}

export interface AutomationStatus {
  enabled: boolean;
  dataset_path: string | null;
  output_path: string;
  state_path: string;
  interval_minutes: number;
  default_zone_id: string;
  video_frame_step: number;
  recursive: boolean;
  reprocess_completed: boolean;
  tracked_items: number;
  item_status_counts: Record<string, number>;
  running: boolean;
  job_running: boolean;
  last_processed_file: string | null;
  last_processed_frame: number | null;
  last_processed_zone: string | null;
  last_processed_at: string | null;
  last_result_path: string | null;
  last_error: unknown;
  last_job: unknown;
}

export const api = {
  getTelemetry: async () => {
    const res = await fetch(`${API_BASE}/telemetry`);
    return res.json();
  },
  getRecommendations: async () => {
    const res = await fetch(`${API_BASE}/recommendations`);
    return res.json();
  },
  uploadSnapshot: async (zoneId: string, file: File): Promise<SnapshotResult> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/snapshots/${zoneId}`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Snapshot detection failed');
    return res.json();
  },
  getSnapshot: async (zoneId: string) => {
    const res = await fetch(`${API_BASE}/snapshots/${zoneId}`);
    return res.json();
  },
  getAutomationStatus: async (): Promise<AutomationStatus> => {
    const res = await fetch(`${API_BASE}/automation/status`);
    return res.json();
  },
  runAutomationNow: async () => {
    const res = await fetch(`${API_BASE}/automation/run-now`, { method: 'POST' });
    return res.json();
  },
  overrideSetpoint: async (zoneId: string, setpoint: number, isManual = true) => {
    const res = await fetch(`${API_BASE}/bacnet/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zone_id: zoneId, setpoint_c: setpoint, is_manual: isManual }),
    });
    return res.json();
  },
  applyAllRecommendations: async () => {
    const res = await fetch(`${API_BASE}/bacnet/apply-all-recommendations`, { method: 'POST' });
    return res.json();
  },
  simulateStep: async (minutes = 1) => {
    const res = await fetch(`${API_BASE}/simulate/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ minutes }),
    });
    return res.json();
  },
  chatAI: async (message: string) => {
    const res = await fetch(`${API_BASE}/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    return res.json();
  },
};
