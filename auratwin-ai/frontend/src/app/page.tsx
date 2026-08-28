'use client';

import {
  Activity,
  AirVent,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BatteryCharging,
  Bell,
  Bot,
  BrainCircuit,
  CalendarClock,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Clock3,
  CloudCog,
  Cpu,
  Gauge,
  Info,
  LayoutDashboard,
  Loader2,
  MessageCircle,
  Moon,
  Play,
  RefreshCw,
  Send,
  Settings2,
  SlidersHorizontal,
  Thermometer,
  Upload,
  Users,
  Zap,
} from 'lucide-react';
import type { FormEvent, ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { api, type AIRecommendation, type AutomationStatus, type EnergySummary, type SnapshotResult, type ZoneTelemetry } from '../lib/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type TelemetryPayload = {
  simulation_time?: string;
  time_str?: string;
  telemetry: ZoneTelemetry[];
  energy_summary: EnergySummary;
};

type RecommendationPayload = {
  tariff_tier?: string;
  price_per_kwh?: number;
  total_recommendations?: number;
  total_potential_savings_kw?: number;
  recommendations: AIRecommendation[];
};

type PowerPoint = {
  time: string;
  current: number;
  baseline: number;
};

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

const EMPTY_ENERGY: EnergySummary = {
  current_power_kw: 0,
  baseline_power_kw: 0,
  predicted_power_kw: 0,
  potential_saving_kw: 0,
  saving_percentage: 0,
  tariff_tier: 'Standard',
  price_per_kwh: 0,
  time_window: '--',
  hourly_cost_usd: 0,
  hourly_savings_usd: 0,
  today_consumption_kwh: 0,
  today_savings_usd: 0,
};

const INITIAL_CHAT: ChatMessage = {
  role: 'assistant',
  content: 'Welcome to AuraTwin AI. I can explain zone conditions, energy waste, savings, and K-Means density insights using the live digital twin.',
};

function numberValue(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function formatNumber(value: unknown, digits = 1): string {
  return numberValue(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatCurrency(value: unknown): string {
  return `$${numberValue(value).toFixed(2)}`;
}

function formatDate(value: unknown): string {
  if (!value || typeof value !== 'string') return 'Not available';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function safeText(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return fallback;
  try {
    return JSON.stringify(value);
  } catch {
    return fallback;
  }
}

function stripMarkdown(value: string): string {
  return value.replace(/\*\*/g, '').replace(/`/g, '');
}

function zoneTemperature(zone: ZoneTelemetry | undefined): { fill: string; stroke: string; label: string } {
  const temperature = numberValue(zone?.temperature_c, 24);
  if (temperature >= 28) return { fill: '#7f1d1d', stroke: '#f87171', label: 'Hot' };
  if (temperature >= 26) return { fill: '#78350f', stroke: '#fbbf24', label: 'Warm' };
  if (temperature >= 22) return { fill: '#164e63', stroke: '#22d3ee', label: 'Optimal' };
  return { fill: '#172554', stroke: '#818cf8', label: 'Cool' };
}

function displayZoneType(value: string | undefined): string {
  return (value || 'zone').replace(/_/g, ' ');
}

function MetricCard({ icon, label, value, detail, tone = 'cyan' }: { icon: ReactNode; label: string; value: string; detail: string; tone?: 'cyan' | 'emerald' | 'purple' | 'amber' }) {
  const tones = {
    cyan: 'border-cyan-400/10 bg-cyan-400/[0.04] text-cyan-300',
    emerald: 'border-emerald-400/10 bg-emerald-400/[0.04] text-emerald-300',
    purple: 'border-purple-400/10 bg-purple-400/[0.04] text-purple-300',
    amber: 'border-amber-400/10 bg-amber-400/[0.04] text-amber-300',
  };

  return (
    <div className="glass-card group relative overflow-hidden p-4 transition hover:-translate-y-0.5 hover:border-white/15">
      <div className={`mb-5 flex h-9 w-9 items-center justify-center rounded-xl border ${tones[tone]}`}>{icon}</div>
      <p className="label">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight text-white">{value}</p>
      <p className="mt-1 truncate text-[11px] text-slate-500">{detail}</p>
      <div className="absolute -right-8 -top-8 h-24 w-24 rounded-full bg-white/[0.02] blur-2xl transition group-hover:bg-cyan-300/[0.06]" />
    </div>
  );
}

function SectionHeading({ icon, eyebrow, title, action }: { icon: ReactNode; eyebrow: string; title: string; action?: ReactNode }) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-cyan-300">{icon}</div>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="mt-1 text-base font-semibold text-white">{title}</h2>
        </div>
      </div>
      {action}
    </div>
  );
}

function StatusDot({ active = true, label }: { active?: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-slate-300">
      <span className={`h-2 w-2 rounded-full ${active ? 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]' : 'bg-slate-600'}`} />
      {label}
    </span>
  );
}

export default function DashboardPage() {
  const [telemetryData, setTelemetryData] = useState<TelemetryPayload | null>(null);
  const [recommendationData, setRecommendationData] = useState<RecommendationPayload | null>(null);
  const [automation, setAutomation] = useState<AutomationStatus | null>(null);
  const [selectedZoneId, setSelectedZoneId] = useState('');
  const [powerHistory, setPowerHistory] = useState<PowerPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoSimulation, setAutoSimulation] = useState(false);
  const [simulationBusy, setSimulationBusy] = useState(false);
  const [actionMessage, setActionMessage] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [snapshotResult, setSnapshotResult] = useState<SnapshotResult | null>(null);
  const [snapshotUrl, setSnapshotUrl] = useState('');
  const [manualSetpoint, setManualSetpoint] = useState(23);
  const [overrideBusy, setOverrideBusy] = useState(false);
  const [applyAllBusy, setApplyAllBusy] = useState(false);
  const [automationBusy, setAutomationBusy] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([INITIAL_CHAT]);
  const [chatInput, setChatInput] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const simulationLock = useRef(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const refreshDashboard = useCallback(async (quiet = false) => {
    if (quiet) setRefreshing(true);
    else setLoading(true);

    const [telemetryResult, recommendationResult, automationResult] = await Promise.allSettled([
      api.getTelemetry(),
      api.getRecommendations(),
      api.getAutomationStatus(),
    ]);

    const errors: string[] = [];
    if (telemetryResult.status === 'fulfilled') {
      const value = telemetryResult.value as TelemetryPayload;
      if (Array.isArray(value?.telemetry)) {
        setTelemetryData(value);
        const summary = value.energy_summary || EMPTY_ENERGY;
        setPowerHistory((previous) => {
          const point: PowerPoint = {
            time: value.time_str || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            current: numberValue(summary.current_power_kw),
            baseline: numberValue(summary.baseline_power_kw),
          };
          const next = [...previous, point];
          return next.slice(-24);
        });
      } else {
        errors.push('The telemetry response was empty.');
      }
    } else {
      errors.push('Backend telemetry is unavailable. Start FastAPI on port 8000.');
    }

    if (recommendationResult.status === 'fulfilled') {
      const value = recommendationResult.value as RecommendationPayload;
      setRecommendationData({ ...value, recommendations: Array.isArray(value?.recommendations) ? value.recommendations : [] });
    } else {
      errors.push('HVAC recommendations could not be loaded.');
    }

    if (automationResult.status === 'fulfilled') {
      setAutomation(automationResult.value as AutomationStatus);
    } else {
      errors.push('Automation status could not be loaded.');
    }

    setError(errors.join(' '));
    setLastUpdated(new Date());
    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => {
    void refreshDashboard();
    const interval = window.setInterval(() => void refreshDashboard(true), 10000);
    return () => window.clearInterval(interval);
  }, [refreshDashboard]);

  const zones = telemetryData?.telemetry || [];
  const energy = telemetryData?.energy_summary || EMPTY_ENERGY;
  const recommendations = recommendationData?.recommendations || [];
  const selectedZone = zones.find((zone) => zone.zone_id === selectedZoneId) || zones[0];
  const selectedRecommendation = recommendations.find((recommendation) => recommendation.zone_id === selectedZone?.zone_id);

  useEffect(() => {
    if (!selectedZoneId && zones[0]) setSelectedZoneId(zones[0].zone_id);
    if (selectedZoneId && zones.length > 0 && !zones.some((zone) => zone.zone_id === selectedZoneId)) setSelectedZoneId(zones[0].zone_id);
  }, [selectedZoneId, zones]);

  useEffect(() => {
    if (selectedZone) setManualSetpoint(numberValue(selectedZone.setpoint_c, 23));
  }, [selectedZone]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, chatBusy]);

  const runSimulation = useCallback(async (minutes: number) => {
    if (simulationLock.current) return;
    simulationLock.current = true;
    setSimulationBusy(true);
    setActionMessage(`Advancing the virtual clock by ${minutes} minute${minutes === 1 ? '' : 's'}…`);
    try {
      await api.simulateStep(minutes);
      await refreshDashboard(true);
      setActionMessage(`Simulation advanced by ${minutes} minute${minutes === 1 ? '' : 's'}.`);
    } catch {
      setActionMessage('Simulation could not be advanced. Check that the backend is running.');
    } finally {
      simulationLock.current = false;
      setSimulationBusy(false);
    }
  }, [refreshDashboard]);

  useEffect(() => {
    if (!autoSimulation) return;
    const interval = window.setInterval(() => void runSimulation(1), 15000);
    return () => window.clearInterval(interval);
  }, [autoSimulation, runSimulation]);

  const handleUpload = async () => {
    if (!uploadFile || !selectedZone) {
      setActionMessage('Choose an image and a zone before running detection.');
      return;
    }
    setUploadBusy(true);
    setActionMessage('Running YOLO person detection…');
    try {
      const result = await api.uploadSnapshot(selectedZone.zone_id, uploadFile);
      setSnapshotResult(result);
      const relative = result.annotated_image_url || `/snapshots/${selectedZone.zone_id}/annotated`;
      setSnapshotUrl(`${relative.startsWith('http') ? relative : `${API_BASE}${relative}`}?t=${Date.now()}`);
      await refreshDashboard(true);
      setActionMessage(`${result.people_count} people detected in ${selectedZone.name}.`);
    } catch (uploadError) {
      setActionMessage(safeText(uploadError, 'Snapshot detection failed.'));
    } finally {
      setUploadBusy(false);
    }
  };

  const applyOverride = async () => {
    if (!selectedZone) return;
    setOverrideBusy(true);
    try {
      const result = await api.overrideSetpoint(selectedZone.zone_id, manualSetpoint, true);
      setActionMessage(result?.status === 'applied' ? `Manual setpoint applied to ${selectedZone.name}.` : 'Setpoint command was sent.');
      await refreshDashboard(true);
    } catch {
      setActionMessage('BACnet override failed.');
    } finally {
      setOverrideBusy(false);
    }
  };

  const applyRecommendation = async (recommendation: AIRecommendation) => {
    try {
      await api.overrideSetpoint(recommendation.zone_id, recommendation.recommended_setpoint_c, false);
      setActionMessage(`AI setpoint applied to ${recommendation.zone_name}.`);
      await refreshDashboard(true);
    } catch {
      setActionMessage('Could not apply that recommendation.');
    }
  };

  const applyAllRecommendations = async () => {
    setApplyAllBusy(true);
    try {
      const result = await api.applyAllRecommendations();
      setActionMessage(`${numberValue(result?.count, recommendations.length)} AI setpoints applied.`);
      await refreshDashboard(true);
    } catch {
      setActionMessage('Could not apply all recommendations.');
    } finally {
      setApplyAllBusy(false);
    }
  };

  const runAutomation = async () => {
    setAutomationBusy(true);
    setActionMessage('Running the next CCTV dataset item…');
    try {
      const result = await api.runAutomationNow();
      setActionMessage(result?.status === 'completed' ? 'CCTV automation job completed.' : safeText(result?.message, 'Automation job finished.'));
      await refreshDashboard(true);
    } catch {
      setActionMessage('The automation job could not run.');
    } finally {
      setAutomationBusy(false);
    }
  };

  const submitChat = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = chatInput.trim();
    if (!message || chatBusy) return;
    setChatInput('');
    setChatMessages((previous) => [...previous, { role: 'user', content: message }]);
    setChatBusy(true);
    try {
      const result = await api.chatAI(message);
      const content = result?.response || result?.message || 'The facility manager did not return a response.';
      setChatMessages((previous) => [...previous, { role: 'assistant', content: String(content) }]);
    } catch {
      setChatMessages((previous) => [...previous, { role: 'assistant', content: 'I could not reach the facility manager service. Please check the backend connection.' }]);
    } finally {
      setChatBusy(false);
    }
  };

  const annotatedImage = snapshotUrl || (selectedZone?.snapshot_uploaded ? `${API_BASE}/snapshots/${selectedZone.zone_id}/annotated` : '');
  const totalOccupancy = zones.reduce((sum, zone) => sum + numberValue(zone.estimated_occupancy), 0);
  const detectedPeople = zones.reduce((sum, zone) => sum + numberValue(zone.detected_people), 0);
  const totalCapacity = zones.reduce((sum, zone) => sum + numberValue(zone.capacity), 0);
  const occupancyPercent = totalCapacity ? (totalOccupancy / totalCapacity) * 100 : 0;
  const simulationTime = telemetryData?.time_str || '--:--:--';
  const tariff = recommendationData?.tariff_tier || energy.tariff_tier || 'Standard';
  const selectedTone = zoneTemperature(selectedZone);

  return (
    <div className="min-h-screen pb-10">
      <header className="sticky top-0 z-30 border-b border-white/[0.07] bg-[#070b12]/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="relative flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-300/25 bg-cyan-400/10 text-cyan-300 shadow-lg shadow-cyan-950/30">
              <BrainCircuit size={23} />
              <span className="absolute -bottom-1 -right-1 h-3 w-3 rounded-full border-2 border-[#070b12] bg-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-white sm:text-lg">AuraTwin <span className="text-cyan-300">AI</span></h1>
                <span className="hidden rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.16em] text-slate-400 sm:inline">Live</span>
              </div>
              <p className="text-[11px] text-slate-500">10-Zone Digital Twin · HVAC Intelligence</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2 sm:gap-3">
            <div className="hidden items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.03] px-3 py-2 md:flex">
              <StatusDot active label="Virtual BACnet online" />
              <span className="h-4 w-px bg-white/10" />
              <span className="inline-flex items-center gap-1.5 text-xs text-slate-300"><Clock3 size={13} className="text-cyan-300" /> {simulationTime}</span>
            </div>
            <button type="button" className="secondary-button" onClick={() => void runSimulation(1)} disabled={simulationBusy}><Clock3 size={14} /> +1 Min</button>
            <button type="button" className="secondary-button hidden sm:inline-flex" onClick={() => void runSimulation(15)} disabled={simulationBusy}><Clock3 size={14} /> +15 Min</button>
            <button type="button" className={`secondary-button ${autoSimulation ? 'border-emerald-300/30 bg-emerald-400/10 text-emerald-200' : ''}`} onClick={() => setAutoSimulation((value) => !value)}><Play size={13} fill={autoSimulation ? 'currentColor' : 'none'} /> {autoSimulation ? 'Auto Sim On' : 'Auto Simulation'}</button>
            <button type="button" className="primary-button hidden lg:inline-flex" onClick={() => void applyAllRecommendations()} disabled={applyAllBusy || recommendations.length === 0}><Zap size={14} /> {applyAllBusy ? 'Applying…' : 'Auto-Apply AI Setpoints'}</button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] space-y-5 px-4 pt-6 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="eyebrow">Operations overview</p>
            <h2 className="mt-1 text-xl font-semibold tracking-tight text-white sm:text-2xl">Building command center</h2>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            {lastUpdated && <span>Updated {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>}
            <button type="button" onClick={() => void refreshDashboard(true)} className="secondary-button !px-2.5 !py-2" disabled={refreshing}><RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} /></button>
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-3 rounded-xl border border-amber-300/20 bg-amber-300/[0.06] px-4 py-3 text-sm text-amber-100">
            <AlertTriangle size={17} className="mt-0.5 shrink-0 text-amber-300" />
            <div className="flex-1"><p className="font-semibold">Some live data is unavailable</p><p className="mt-0.5 text-xs text-amber-100/70">{error}</p></div>
            <button type="button" className="text-xs font-semibold text-amber-200 underline underline-offset-4" onClick={() => void refreshDashboard()}>Retry</button>
          </div>
        )}

        {actionMessage && <div className="flex items-center gap-2 rounded-xl border border-cyan-300/15 bg-cyan-300/[0.05] px-4 py-2.5 text-xs text-cyan-100"><Info size={15} className="text-cyan-300" />{actionMessage}</div>}

        <section className="grid grid-cols-2 gap-3 lg:grid-cols-4 xl:grid-cols-8">
          <MetricCard icon={<Zap size={18} />} label="Building power" value={`${formatNumber(energy.current_power_kw)} kW`} detail={`Baseline ${formatNumber(energy.baseline_power_kw)} kW`} tone="cyan" />
          <MetricCard icon={<Activity size={18} />} label="Baseline power" value={`${formatNumber(energy.baseline_power_kw)} kW`} detail={`${formatNumber(energy.potential_saving_kw)} kW potential`} tone="amber" />
          <MetricCard icon={<ArrowDownRight size={18} />} label="Potential savings" value={`${formatNumber(energy.potential_saving_kw)} kW`} detail={`${formatNumber(energy.saving_percentage)}% below baseline`} tone="emerald" />
          <MetricCard icon={<CircleGauge size={18} />} label="Active tariff" value={tariff} detail={`${formatCurrency(energy.price_per_kwh)} per kWh`} tone="amber" />
          <MetricCard icon={<Users size={18} />} label="Estimated occupancy" value={formatNumber(totalOccupancy, 0)} detail={`${formatNumber(occupancyPercent)}% of ${formatNumber(totalCapacity, 0)} capacity`} tone="purple" />
          <MetricCard icon={<Gauge size={18} />} label="CCTV people" value={formatNumber(detectedPeople, 0)} detail="Across monitored zones" tone="purple" />
          <MetricCard icon={<BatteryCharging size={18} />} label="Today’s energy saved" value={formatCurrency(energy.today_savings_usd)} detail={`${formatCurrency(energy.hourly_savings_usd)} hourly run-rate`} tone="emerald" />
          <MetricCard icon={<CalendarClock size={18} />} label="Today’s consumption" value={`${formatNumber(energy.today_consumption_kwh)} kWh`} detail={energy.time_window || 'Live simulation'} tone="cyan" />
        </section>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1.42fr)_minmax(360px,0.58fr)]">
          <div className="glass-card min-w-0 p-5 sm:p-6">
            <SectionHeading icon={<LayoutDashboard size={18} />} eyebrow="Spatial intelligence" title="2D building floorplan" action={<div className="hidden items-center gap-3 text-[10px] text-slate-500 sm:flex"><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-cyan-300" /> Optimal</span><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-amber-300" /> Warm</span><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-red-400" /> Hot</span></div>} />
            <div className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0a111e] p-2 sm:p-4">
              <div className="pointer-events-none absolute inset-0 opacity-30" style={{ backgroundImage: 'linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px)', backgroundSize: '34px 34px' }} />
              {loading && zones.length === 0 ? (
                <div className="flex h-[390px] items-center justify-center text-sm text-slate-500"><Loader2 size={18} className="mr-2 animate-spin" /> Loading floorplan telemetry…</div>
              ) : zones.length === 0 ? (
                <div className="flex h-[390px] items-center justify-center text-sm text-slate-500">No zones returned by the backend.</div>
              ) : (
                <svg viewBox="0 0 920 540" className="relative z-10 h-auto max-h-[500px] w-full" role="img" aria-label="Interactive building floorplan">
                  <rect x="22" y="22" width="876" height="496" rx="16" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="2" strokeDasharray="5 7" />
                  <text x="46" y="43" fill="rgba(148,163,184,0.55)" fontSize="10" letterSpacing="2">NORTH WING · LIVE THERMAL VIEW</text>
                  {zones.map((zone) => {
                    const floor = zone.floorplan || { x: 50, y: 50, width: 150, height: 100 };
                    const tone = zoneTemperature(zone);
                    const isSelected = zone.zone_id === selectedZone?.zone_id;
                    const centerX = floor.x + floor.width / 2;
                    const centerY = floor.y + floor.height / 2;
                    return (
                      <g key={zone.zone_id} onClick={() => setSelectedZoneId(zone.zone_id)} className="cursor-pointer">
                        <title>{`${zone.name}: ${formatNumber(zone.temperature_c)}°C, ${formatNumber(zone.detected_people, 0)} people`}</title>
                        <rect x={floor.x} y={floor.y} width={floor.width} height={floor.height} rx="10" fill={tone.fill} fillOpacity={isSelected ? 0.95 : 0.74} stroke={isSelected ? '#f8fafc' : tone.stroke} strokeWidth={isSelected ? 3 : 1.5} strokeOpacity={isSelected ? 1 : 0.72} />
                        {isSelected && <rect x={floor.x - 5} y={floor.y - 5} width={floor.width + 10} height={floor.height + 10} rx="14" fill="none" stroke="#22d3ee" strokeOpacity="0.45" strokeDasharray="4 5" />}
                        <text x={centerX} y={centerY - 15} textAnchor="middle" fill="#f8fafc" fontSize={floor.width < 150 ? 11 : 13} fontWeight="600">{zone.name}</text>
                        <text x={centerX} y={centerY + 5} textAnchor="middle" fill="rgba(226,232,240,0.86)" fontSize="11">{formatNumber(zone.temperature_c)}°C · {formatNumber(zone.detected_people, 0)} ppl</text>
                        <text x={centerX} y={centerY + 23} textAnchor="middle" fill={tone.stroke} fontSize="10" fontWeight="600">{tone.label} · {zone.density_cluster || '—'}</text>
                      </g>
                    );
                  })}
                  <path d="M32 336H888" stroke="rgba(148,163,184,0.16)" strokeWidth="2" strokeDasharray="8 8" />
                  <text x="54" y="510" fill="rgba(148,163,184,0.5)" fontSize="10" letterSpacing="1.5">SOUTH WING · CLICK A ZONE FOR DETAILS</text>
                </svg>
              )}
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Selected zone</p><p className="mt-1 truncate font-semibold text-white">{selectedZone?.name || '—'}</p></div>
              <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Density index</p><p className="mt-1 font-semibold text-cyan-200">{selectedZone ? formatNumber(selectedZone.zdi, 2) : '—'}</p></div>
              <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">HVAC status</p><p className="mt-1 font-semibold text-emerald-200">{selectedZone?.hvac_status || '—'}</p></div>
              <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Manual override</p><p className="mt-1 font-semibold text-amber-200">{selectedZone?.manual_override ? 'Active' : 'Inactive'}</p></div>
            </div>
          </div>

          <div className="space-y-5">
            <div className="glass-card p-5">
              <SectionHeading icon={<Thermometer size={18} />} eyebrow="Zone detail" title={selectedZone?.name || 'Select a zone'} action={selectedZone && <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider ${selectedTone.label === 'Hot' ? 'border-red-300/20 bg-red-400/10 text-red-200' : selectedTone.label === 'Warm' ? 'border-amber-300/20 bg-amber-400/10 text-amber-200' : 'border-cyan-300/20 bg-cyan-400/10 text-cyan-200'}`}>{selectedTone.label}</span>} />
              {selectedZone ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Temperature</p><p className="mt-1 text-xl font-semibold text-white">{formatNumber(selectedZone.temperature_c)}°C</p><p className="mt-0.5 text-[11px] text-slate-500">Humidity {formatNumber(selectedZone.humidity_pct)}%</p></div>
                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Power draw</p><p className="mt-1 text-xl font-semibold text-white">{formatNumber(selectedZone.power_kw)} kW</p><p className="mt-0.5 text-[11px] text-slate-500">Baseline {formatNumber(selectedZone.baseline_power_kw)} kW</p></div>
                  </div>
                  <div><div className="mb-2 flex items-center justify-between text-xs"><span className="text-slate-400">Occupancy</span><span className="font-semibold text-purple-200">{formatNumber(selectedZone.estimated_occupancy, 0)} / {formatNumber(selectedZone.capacity, 0)} people</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-purple-500 to-fuchsia-400 transition-all" style={{ width: `${Math.min(100, numberValue(selectedZone.occupancy_percentage))}%` }} /></div><div className="mt-1 flex justify-between text-[10px] text-slate-500"><span>{selectedZone.occupancy_category || 'Unclassified'}</span><span>{formatNumber(selectedZone.occupancy_percentage)}%</span></div></div>
                  <div className="grid grid-cols-3 gap-2 text-center"><div className="rounded-xl bg-white/[0.03] p-2"><p className="text-sm font-semibold text-cyan-200">{formatNumber(selectedZone.setpoint_c)}°</p><p className="text-[10px] text-slate-500">Setpoint</p></div><div className="rounded-xl bg-white/[0.03] p-2"><p className="text-sm font-semibold text-purple-200">{formatNumber(selectedZone.detected_people, 0)}</p><p className="text-[10px] text-slate-500">CCTV people</p></div><div className="rounded-xl bg-white/[0.03] p-2"><p className="truncate text-sm font-semibold text-emerald-200">{selectedZone.density_cluster || '—'}</p><p className="text-[10px] text-slate-500">K-Means</p></div></div>
                </div>
              ) : <p className="text-sm text-slate-500">Zone information will appear after telemetry loads.</p>}
            </div>

            <div className="glass-card p-5">
              <SectionHeading icon={<SlidersHorizontal size={18} />} eyebrow="BACnet control" title="Manual setpoint" action={<span className="text-[10px] uppercase tracking-wider text-slate-500">Virtual AV:1</span>} />
              {selectedZone ? <div className="space-y-4"><div className="flex items-end justify-between"><div><p className="label">Current setpoint</p><p className="mt-1 text-2xl font-semibold text-white">{formatNumber(selectedZone.setpoint_c)}°C</p></div><span className="rounded-full border border-emerald-300/20 bg-emerald-400/10 px-2.5 py-1 text-[10px] font-semibold text-emerald-200">{selectedZone.manual_override ? 'MANUAL OVERRIDE' : 'AI / NORMAL'}</span></div><input aria-label="Manual temperature setpoint" type="range" min="18" max="28" step="0.5" value={manualSetpoint} onChange={(event) => setManualSetpoint(Number(event.target.value))} className="w-full accent-cyan-400" /><div className="flex justify-between text-[10px] text-slate-500"><span>18°C</span><span className="font-semibold text-cyan-200">Selected {formatNumber(manualSetpoint)}°C</span><span>28°C</span></div><button type="button" className="primary-button w-full" onClick={() => void applyOverride()} disabled={overrideBusy}>{overrideBusy ? <Loader2 size={14} className="animate-spin" /> : <Settings2 size={14} />} {overrideBusy ? 'Writing BACnet value…' : 'Apply manual override'}</button></div> : <p className="text-sm text-slate-500">Select a zone to control its virtual thermostat.</p>}
            </div>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <div className="glass-card p-5 sm:p-6">
            <SectionHeading icon={<CloudCog size={18} />} eyebrow="Decision engine" title="HVAC optimization" action={<button type="button" className="primary-button !px-2.5" onClick={() => void applyAllRecommendations()} disabled={applyAllBusy || recommendations.length === 0}>{applyAllBusy ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />} Apply all</button>} />
            {selectedRecommendation ? <div className="mb-4 rounded-xl border border-cyan-300/15 bg-cyan-300/[0.05] p-3"><div className="flex items-start justify-between gap-3"><div><p className="text-sm font-semibold text-white">{selectedRecommendation.zone_name}</p><p className="mt-0.5 text-xs capitalize text-slate-500">{selectedRecommendation.mode} · {selectedRecommendation.tariff_adjusted ? 'tariff adjusted' : 'standard tariff'}</p></div><span className="rounded-lg bg-emerald-400/10 px-2 py-1 text-xs font-semibold text-emerald-200">−{formatNumber(selectedRecommendation.expected_saving_kw)} kW</span></div><div className="mt-3 grid grid-cols-2 gap-2 text-xs"><div className="rounded-lg bg-black/15 p-2.5"><p className="text-slate-500">Current</p><p className="mt-1 font-semibold text-white">{formatNumber(selectedRecommendation.current_setpoint_c)}°C</p></div><div className="rounded-lg bg-black/15 p-2.5"><p className="text-slate-500">Recommended</p><p className="mt-1 font-semibold text-cyan-200">{formatNumber(selectedRecommendation.recommended_setpoint_c)}°C</p></div></div><p className="mt-3 text-xs leading-relaxed text-slate-400">{selectedRecommendation.reason}</p><button type="button" className="primary-button mt-3 w-full" onClick={() => void applyRecommendation(selectedRecommendation)}><ArrowUpRight size={14} /> Apply for {selectedRecommendation.zone_name}</button></div> : <div className="mb-4 rounded-xl border border-dashed border-white/10 p-4 text-sm text-slate-500">No recommendation is available for the selected zone.</div>}
            <div className="max-h-56 space-y-2 overflow-y-auto pr-1">{recommendations.length === 0 ? <p className="py-4 text-center text-xs text-slate-500">No recommendation data returned.</p> : recommendations.map((recommendation) => <button type="button" key={recommendation.zone_id} onClick={() => setSelectedZoneId(recommendation.zone_id)} className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition ${recommendation.zone_id === selectedZone?.zone_id ? 'border-cyan-300/25 bg-cyan-300/[0.06]' : 'border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.05]'}`}><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-[10px] font-bold text-slate-300">{recommendation.zone_id.replace('zone_', '')}</span><span className="min-w-0 flex-1"><span className="block truncate text-xs font-semibold text-slate-200">{recommendation.zone_name}</span><span className="block truncate text-[10px] text-slate-500">{recommendation.mode} · target {formatNumber(recommendation.recommended_setpoint_c)}°C</span></span><span className="text-[11px] font-semibold text-emerald-300">−{formatNumber(recommendation.expected_saving_kw)} kW</span><ChevronRight size={14} className="text-slate-600" /></button>)}</div>
          </div>

          <div className="glass-card p-5 sm:p-6">
            <SectionHeading icon={<Upload size={18} />} eyebrow="Computer vision" title="CCTV snapshot detection" action={<span className="rounded-full border border-purple-300/20 bg-purple-400/10 px-2.5 py-1 text-[10px] font-semibold text-purple-200">YOLO person count</span>} />
            <div className="grid gap-5 md:grid-cols-[minmax(0,0.8fr)_minmax(220px,1.2fr)]">
              <div className="space-y-4">
                <label className="block"><span className="label">Selected zone</span><select value={selectedZone?.zone_id || ''} onChange={(event) => setSelectedZoneId(event.target.value)} className="mt-2 w-full rounded-xl border border-white/10 bg-slate-950/70 px-3 py-2.5 text-sm text-slate-200"><option value="" disabled>Choose a zone</option>{zones.map((zone) => <option key={zone.zone_id} value={zone.zone_id}>{zone.name}</option>)}</select></label>
                <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-dashed border-purple-300/20 bg-purple-400/[0.04] p-3 transition hover:border-purple-300/40"><input type="file" accept="image/*" className="sr-only" onChange={(event) => setUploadFile(event.target.files?.[0] || null)} /><span className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-400/10 text-purple-300"><Upload size={17} /></span><span className="min-w-0"><span className="block truncate text-xs font-semibold text-slate-200">{uploadFile?.name || 'Choose CCTV image'}</span><span className="block text-[10px] text-slate-500">JPEG, PNG or OpenCV-readable image</span></span></label>
                <button type="button" className="primary-button w-full !border-purple-300/20 !bg-purple-400/10 !text-purple-100 hover:!bg-purple-400/20" onClick={() => void handleUpload()} disabled={uploadBusy || !uploadFile || !selectedZone}>{uploadBusy ? <Loader2 size={14} className="animate-spin" /> : <Cpu size={14} />} {uploadBusy ? 'Detecting people…' : 'Detect people'}</button>
                <div className="grid grid-cols-2 gap-2"> <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">People count</p><p className="mt-1 text-2xl font-semibold text-purple-200">{snapshotResult ? snapshotResult.people_count : formatNumber(selectedZone?.detected_people, 0)}</p></div><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Head count</p><p className="mt-1 text-2xl font-semibold text-fuchsia-200">{snapshotResult ? snapshotResult.head_count : formatNumber(selectedZone?.detected_people, 0)}</p></div></div>
              </div>
              <div className="flex min-h-[220px] items-center justify-center overflow-hidden rounded-2xl border border-white/[0.08] bg-slate-950/70">{annotatedImage ? <img src={annotatedImage} alt="Latest annotated CCTV snapshot" className="h-full max-h-[300px] w-full object-contain" /> : <div className="px-6 text-center"><div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-purple-400/10 text-purple-300"><Users size={22} /></div><p className="mt-3 text-sm font-semibold text-slate-300">No annotated snapshot yet</p><p className="mt-1 text-xs leading-relaxed text-slate-500">Upload a frame to see the detected people and bounding boxes here.</p></div>}</div>
            </div>
          </div>
        </section>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(330px,0.6fr)]">
          <div className="glass-card min-w-0 p-5 sm:p-6">
            <SectionHeading icon={<Activity size={18} />} eyebrow="Energy analytics" title="Power versus baseline" action={<div className="flex items-center gap-3 text-[10px] text-slate-500"><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-cyan-300" /> Current / optimized</span><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-slate-500" /> Baseline</span></div>} />
            <div className="h-[280px] w-full">{powerHistory.length < 1 ? <div className="flex h-full items-center justify-center text-sm text-slate-500"><Loader2 size={17} className="mr-2 animate-spin" /> Building live chart…</div> : <ResponsiveContainer width="100%" height="100%"><AreaChart data={powerHistory} margin={{ top: 8, right: 5, left: -20, bottom: 0 }}><defs><linearGradient id="currentPower" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#22d3ee" stopOpacity={0.32} /><stop offset="100%" stopColor="#22d3ee" stopOpacity={0} /></linearGradient><linearGradient id="baselinePower" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#64748b" stopOpacity={0.16} /><stop offset="100%" stopColor="#64748b" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="rgba(148,163,184,0.09)" vertical={false} /><XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} minTickGap={26} /><YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} width={42} /><Tooltip contentStyle={{ background: '#111a2b', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 12, color: '#e2e8f0', fontSize: 12 }} labelStyle={{ color: '#94a3b8' }} formatter={(value: number) => [`${formatNumber(value)} kW`, '']} /><Area type="monotone" dataKey="baseline" stroke="#64748b" strokeWidth={1.5} strokeDasharray="5 5" fill="url(#baselinePower)" name="Baseline" /><Area type="monotone" dataKey="current" stroke="#22d3ee" strokeWidth={2.5} fill="url(#currentPower)" name="Current" /></AreaChart></ResponsiveContainer>}</div>
            <div className="mt-4 grid grid-cols-3 gap-3"><div><p className="label">Current</p><p className="mt-1 text-sm font-semibold text-cyan-200">{formatNumber(energy.current_power_kw)} kW</p></div><div><p className="label">Predicted optimized</p><p className="mt-1 text-sm font-semibold text-emerald-200">{formatNumber(energy.predicted_power_kw)} kW</p></div><div><p className="label">Hourly cost</p><p className="mt-1 text-sm font-semibold text-white">{formatCurrency(energy.hourly_cost_usd)}</p></div></div>
          </div>

          <div className="glass-card p-5 sm:p-6">
            <SectionHeading icon={<CloudCog size={18} />} eyebrow="Dataset scheduler" title="CCTV automation" action={<StatusDot active={Boolean(automation?.enabled)} label={automation?.enabled ? 'Enabled' : 'Disabled'} />} />
            <div className="grid grid-cols-2 gap-2 text-xs"><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Schedule</p><p className="mt-1 font-semibold text-white">Every {formatNumber(automation?.interval_minutes, 0)} min</p></div><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3"><p className="label">Job state</p><p className="mt-1 font-semibold text-cyan-200">{automation?.job_running ? 'Processing' : automation?.running ? 'Waiting' : 'Stopped'}</p></div></div>
            <div className="mt-3 space-y-2.5 text-xs"><div className="flex items-start justify-between gap-3"><span className="text-slate-500">Last file</span><span className="max-w-[62%] truncate text-right font-medium text-slate-300" title={safeText(automation?.last_processed_file)}>{automation?.last_processed_file || 'No item processed'}</span></div><div className="flex items-center justify-between gap-3"><span className="text-slate-500">Last zone</span><span className="font-medium text-slate-300">{automation?.last_processed_zone || '—'}</span></div><div className="flex items-center justify-between gap-3"><span className="text-slate-500">Last processed</span><span className="text-right font-medium text-slate-300">{formatDate(automation?.last_processed_at)}</span></div>{Boolean(automation?.last_error) && <div className="rounded-xl border border-red-300/15 bg-red-400/[0.06] p-2.5 text-[11px] text-red-200"><span className="font-semibold">Last error:</span> {safeText(automation?.last_error)}</div>}</div>
            <button type="button" className="primary-button mt-4 w-full" onClick={() => void runAutomation()} disabled={automationBusy || Boolean(automation?.job_running)}>{automationBusy ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} {automationBusy ? 'Processing next item…' : 'Run automation now'}</button>
            <p className="mt-3 text-[10px] leading-relaxed text-slate-500">The backend keeps the cursor in its state file and selects the next image or video frame automatically.</p>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[minmax(0,0.72fr)_minmax(0,1.28fr)]">
          <div className="glass-card p-5 sm:p-6">
            <SectionHeading icon={<Bot size={18} />} eyebrow="Facility intelligence" title="AI facility manager" action={<span className="flex items-center gap-1.5 text-[10px] text-emerald-300"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Data-grounded</span>} />
            <div className="mb-4 flex flex-wrap gap-2"><button type="button" className="secondary-button !px-2.5 !py-1.5 !text-[10px]" onClick={() => setChatInput('Which zone is wasting the most electricity?')}>Energy waste</button><button type="button" className="secondary-button !px-2.5 !py-1.5 !text-[10px]" onClick={() => setChatInput('How much energy did we save today?')}>Savings today</button><button type="button" className="secondary-button !px-2.5 !py-1.5 !text-[10px]" onClick={() => setChatInput('Explain K-Means cluster status')}>K-Means</button></div>
            <div className="flex h-[330px] flex-col rounded-2xl border border-white/[0.08] bg-slate-950/50"><div className="flex-1 space-y-3 overflow-y-auto p-3">{chatMessages.map((message, index) => <div key={`${message.role}-${index}`} className={`flex gap-2.5 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}><div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${message.role === 'user' ? 'bg-cyan-400/10 text-cyan-300' : 'bg-purple-400/10 text-purple-300'}`}>{message.role === 'user' ? <Users size={14} /> : <Bot size={14} />}</div><div className={`max-w-[86%] rounded-2xl px-3 py-2.5 text-xs leading-relaxed ${message.role === 'user' ? 'rounded-tr-sm bg-cyan-400/10 text-cyan-50' : 'rounded-tl-sm border border-white/[0.07] bg-white/[0.04] text-slate-300'}`}><p className="whitespace-pre-wrap">{stripMarkdown(message.content)}</p></div></div>)}{chatBusy && <div className="flex items-center gap-2 text-xs text-slate-500"><Bot size={14} className="text-purple-300" /><Loader2 size={13} className="animate-spin" /> Analyzing live building state…</div>}<div ref={chatEndRef} /></div><form onSubmit={submitChat} className="flex items-center gap-2 border-t border-white/[0.07] p-2.5"><input value={chatInput} onChange={(event) => setChatInput(event.target.value)} placeholder="Ask about your building…" className="min-w-0 flex-1 bg-transparent px-2 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none" /><button type="submit" aria-label="Send message" className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-400 text-slate-950 transition hover:bg-cyan-300 disabled:opacity-40" disabled={chatBusy || !chatInput.trim()}><Send size={14} /></button></form></div>
          </div>

          <div className="glass-card p-5 sm:p-6">
            <SectionHeading icon={<Bell size={18} />} eyebrow="System notes" title="How the digital twin is operating" />
            <div className="grid gap-3 sm:grid-cols-2"><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4"><div className="flex items-center gap-2 text-cyan-200"><CheckCircle2 size={16} /><p className="text-xs font-semibold">Occupancy-aware HVAC</p></div><p className="mt-2 text-xs leading-relaxed text-slate-500">CCTV person counts feed occupancy, density clustering, energy estimates, and setpoint recommendations for every zone.</p></div><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4"><div className="flex items-center gap-2 text-emerald-200"><CalendarClock size={16} /><p className="text-xs font-semibold">Sequential automation</p></div><p className="mt-2 text-xs leading-relaxed text-slate-500">The configured dataset is processed every 20 minutes by the backend scheduler, with a persistent cursor to avoid duplicates.</p></div><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4"><div className="flex items-center gap-2 text-purple-200"><MessageCircle size={16} /><p className="text-xs font-semibold">Ask the manager</p></div><p className="mt-2 text-xs leading-relaxed text-slate-500">Chat answers are generated from the current telemetry, tariff, recommendations, and zone-level conditions.</p></div><div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4"><div className="flex items-center gap-2 text-amber-200"><Moon size={16} /><p className="text-xs font-semibold">Simulation mode</p></div><p className="mt-2 text-xs leading-relaxed text-slate-500">Use the clock controls above to advance virtual thermal physics without touching real HVAC hardware.</p></div></div>
          </div>
        </section>
      </main>
    </div>
  );
}
