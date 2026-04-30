const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchCandidates(phase?: string, section?: string) {
  const params = new URLSearchParams();
  if (phase) params.set("phase", phase);
  if (section) params.set("section", section);
  const res = await fetch(`${BASE}/candidates?${params}`, { cache: "no-store" });
  if (!res.ok) throw Object.assign(new Error("Failed to fetch candidates"), { status: res.status });
  return res.json();
}

export async function approveCandidate(id: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/approve`, { method: "POST" });
  if (!res.ok) throw new Error("Approve failed");
  return res.json();
}

export async function rejectCandidate(id: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/reject`, { method: "POST" });
  if (!res.ok) throw new Error("Reject failed");
  return res.json();
}

export async function setUseCase(id: string, use_case: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/use_case`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_case }),
  });
  if (!res.ok) throw new Error("Use case update failed");
  return res.json();
}

export async function setLabel(id: string, label: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/label`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) throw new Error("Label update failed");
  return res.json();
}

export function midiUrl(id: string) {
  return `${BASE}/midi/${encodeURIComponent(id)}`;
}

export async function promotePhase(phase: string): Promise<{ ok: boolean; promoted_count: number }> {
  const res = await fetch(`${BASE}/promote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phase }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Promote failed");
  }
  return res.json();
}

export async function evolvePhase(phase: string): Promise<{ ok: boolean; evolved_count: number }> {
  const res = await fetch(`${BASE}/evolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phase }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Evolve failed");
  }
  return res.json();
}

export async function fetchSongs(): Promise<import("./types").SongEntry[]> {
  const res = await fetch(`${BASE}/songs`, { cache: "no-store" });
  if (res.status === 503) throw Object.assign(new Error("single-song-mode"), { status: 503 });
  if (!res.ok) throw new Error("Failed to fetch songs");
  return res.json();
}

export async function activateSong(id: string): Promise<{ ok: boolean; production_dir: string }> {
  const res = await fetch(`${BASE}/songs/activate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error("Activate failed");
  return res.json();
}

export async function fetchActiveSong(): Promise<{ active: import("./types").SongEntry | null }> {
  const res = await fetch(`${BASE}/songs/active`, { cache: "no-store" });
  if (!res.ok) return { active: null };
  return res.json();
}

export async function startGenerate(): Promise<{ status: string; started_at: string }> {
  const res = await fetch(`${BASE}/generate`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Generate failed");
  }
  return res.json();
}

export interface GenerateStatus {
  status: "idle" | "running" | "done" | "error";
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export async function getGenerateStatus(): Promise<GenerateStatus> {
  const res = await fetch(`${BASE}/generate/status`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch generate status");
  return res.json();
}

export async function initSong(): Promise<{ ok: boolean; skipped: boolean }> {
  const res = await fetch(`${BASE}/pipeline/init`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Init failed");
  }
  return res.json();
}

export async function runNextPhase(): Promise<{ status: string; started_at: string }> {
  const res = await fetch(`${BASE}/pipeline/run`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Run failed");
  }
  return res.json();
}

export async function getRunStatus(): Promise<import("./types").RunJob> {
  const res = await fetch(`${BASE}/pipeline/run/status`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch run status");
  return res.json();
}

export async function fetchPipelineStatus(): Promise<import("./types").PipelineStatus> {
  const res = await fetch(`${BASE}/pipeline/status`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch pipeline status");
  return res.json();
}

export async function startHandoff(): Promise<{ status: string; started_at: string }> {
  const res = await fetch(`${BASE}/handoff`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Handoff failed");
  }
  return res.json();
}

export async function getHandoffStatus(): Promise<import("./types").RunJob> {
  const res = await fetch(`${BASE}/handoff/status`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch handoff status");
  return res.json();
}

export async function fetchComposition(): Promise<import("./types").CompositionEntry | { status: "not_initialized" }> {
  const res = await fetch(`${BASE}/composition`, { cache: "no-store" });
  if (res.status === 503) return { status: "not_initialized" };
  if (!res.ok) throw new Error("Failed to fetch composition");
  return res.json();
}

export async function advanceStage(stage: string): Promise<{ ok: boolean; stage: string }> {
  const res = await fetch(`${BASE}/composition/stage`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Stage update failed");
  }
  return res.json();
}

export async function addVersion(): Promise<{ ok: boolean; version: number }> {
  const res = await fetch(`${BASE}/composition/version`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Add version failed");
  }
  return res.json();
}
