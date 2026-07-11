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

export async function runQuartetPhase(): Promise<{ status: string; started_at: string }> {
  const res = await fetch(`${BASE}/pipeline/run/quartet`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Quartet run failed");
  }
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

export async function regressStage(
  targetStage: string,
  confirmed: boolean,
  diaryEntry: string | null,
): Promise<import("./types").RegressionInfo | { ok: boolean; stage: string }> {
  const res = await fetch(`${BASE}/composition/regress`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_stage: targetStage, confirmed, diary_entry: diaryEntry }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Regress failed");
  }
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

export async function fetchLyrics(): Promise<import("./types").LyricsResponse> {
  const res = await fetch(`${BASE}/lyrics`, { cache: "no-store" });
  if (res.status === 503) throw Object.assign(new Error("no-active-song"), { status: 503 });
  if (!res.ok) throw Object.assign(new Error("Failed to fetch lyrics"), { status: res.status });
  return res.json();
}

export async function approveLyric(id: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE}/lyrics/${encodeURIComponent(id)}/approve`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Approve lyric failed");
  }
  return res.json();
}

export async function updateVersionNotes(version: number, notes: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE}/composition/version/notes`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version, notes }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Notes update failed");
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

export async function syncArrangement(): Promise<{ ok: boolean; synced_from: string; synced_to: string }> {
  const res = await fetch(`${BASE}/sync-arrangement`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Arrangement sync failed");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Diary
// ---------------------------------------------------------------------------

export async function fetchDiaryEntries(songSlug: string): Promise<import("./types").DiaryEntry[]> {
  const res = await fetch(`${BASE}/diary/${encodeURIComponent(songSlug)}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch diary entries");
  return res.json();
}

export async function createDiaryEntry(
  songSlug: string,
  entry: Omit<import("./types").DiaryEntry, "id" | "created_at">,
): Promise<import("./types").DiaryEntry> {
  const res = await fetch(`${BASE}/diary/${encodeURIComponent(songSlug)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Create diary entry failed");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Collaborator registry
// ---------------------------------------------------------------------------

export async function fetchCollaborators(): Promise<import("./types").Collaborator[]> {
  const res = await fetch(`${BASE}/collaborators`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch collaborators");
  return res.json();
}

export async function fetchCollaborator(id: string): Promise<import("./types").Collaborator> {
  const res = await fetch(`${BASE}/collaborators/${encodeURIComponent(id)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Collaborator '${id}' not found`);
  return res.json();
}

export async function saveCollaborator(c: import("./types").Collaborator): Promise<import("./types").Collaborator> {
  const res = await fetch(`${BASE}/collaborators/${encodeURIComponent(c.id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(c),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Save collaborator failed");
  }
  return res.json();
}

export async function createCollaborator(c: import("./types").Collaborator): Promise<import("./types").Collaborator> {
  const res = await fetch(`${BASE}/collaborators`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(c),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Create collaborator failed");
  }
  return res.json();
}

export async function deleteCollaborator(id: string): Promise<void> {
  const res = await fetch(`${BASE}/collaborators/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok && res.status !== 204) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Delete collaborator failed");
  }
}

// ---------------------------------------------------------------------------
// Work orders
// ---------------------------------------------------------------------------

export async function fetchWorkOrders(): Promise<import("./types").WorkOrder[]> {
  const res = await fetch(`${BASE}/production/work-orders`, { cache: "no-store" });
  if (res.status === 503) return [];
  if (!res.ok) throw new Error("Failed to fetch work orders");
  return res.json();
}

export async function fetchWorkOrder(collaboratorId: string): Promise<import("./types").WorkOrder | null> {
  const res = await fetch(`${BASE}/production/work-orders/${encodeURIComponent(collaboratorId)}`, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error("Failed to fetch work order");
  return res.json();
}

export async function generateWorkOrder(
  collaboratorId: string,
  role: string,
  platform = "direct",
): Promise<import("./types").WorkOrder> {
  const res = await fetch(`${BASE}/production/work-orders/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collaborator_id: collaboratorId, role, platform }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Generate failed");
  }
  return res.json();
}

export async function updateWorkOrder(wo: import("./types").WorkOrder): Promise<import("./types").WorkOrder> {
  const res = await fetch(`${BASE}/production/work-orders/${encodeURIComponent(wo.collaborator_id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(wo),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Save work order failed");
  }
  return res.json();
}

export async function draftWorkOrderEmail(collaboratorId: string): Promise<{
  to: string;
  subject: string;
  body: string;
  status: string;
  note: string;
}> {
  const res = await fetch(`${BASE}/production/work-orders/${encodeURIComponent(collaboratorId)}/draft-email`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Draft email failed");
  }
  return res.json();
}

export interface AutoSplitResult {
  section: string;
  skipped: boolean;
  reason?: string;
  split_midi?: string;
  uncovered_phrase_count?: number;
  warning?: string;
}

export async function autoSplitMelody(): Promise<{
  ok: boolean;
  results: AutoSplitResult[];
  warnings: string[];
}> {
  const res = await fetch(`${BASE}/production/auto-split-melody/all`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Auto-split failed");
  }
  return res.json();
}

export async function assembleMelody(): Promise<{
  ok: boolean;
  assembled_midi: string;
  assembled_lyrics: string | null;
}> {
  const res = await fetch(`${BASE}/production/assemble-melody`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Assemble failed");
  }
  return res.json();
}

export async function setBpm(bpm: number): Promise<{ ok: boolean; bpm: number; updated: string[] }> {
  const res = await fetch(`${BASE}/production/set-bpm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bpm }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "BPM update failed");
  }
  return res.json();
}

export async function fetchMixInfo(): Promise<{ has_mix: boolean; mix_file: string | null }> {
  const res = await fetch(`${BASE}/production/mix/info`);
  if (!res.ok) return { has_mix: false, mix_file: null };
  return res.json();
}

export async function setMixFile(path: string): Promise<void> {
  const res = await fetch(`${BASE}/production/mix/set`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Failed to set mix file");
  }
}

export function mixStreamUrl(): string {
  return `${BASE}/production/mix`;
}

export async function fetchSongMixInfo(
  songId: string,
): Promise<{ has_mix: boolean; mix_file: string | null; duration_seconds: number | null }> {
  const res = await fetch(`${BASE}/songs/${encodeURIComponent(songId)}/mix/info`);
  if (!res.ok) return { has_mix: false, mix_file: null, duration_seconds: null };
  return res.json();
}

export function songMixStreamUrl(songId: string): string {
  return `${BASE}/songs/${encodeURIComponent(songId)}/mix`;
}

export async function fetchSamples(topN?: number): Promise<import("./types").SampleEntry[]> {
  const params = topN != null ? `?top_n=${topN}` : "";
  const res = await fetch(`${BASE}/samples${params}`, { cache: "no-store" });
  if (res.status === 503) throw Object.assign(new Error("no-active-song"), { status: 503 });
  if (!res.ok) throw new Error("Failed to fetch samples");
  return res.json();
}

export function audioUrl(segmentId: string): string {
  return `${BASE}/audio/${encodeURIComponent(segmentId)}`;
}

// ---------------------------------------------------------------------------
// Song lifecycle
// ---------------------------------------------------------------------------

export async function setLifecycleStatus(
  songId: string,
  status: import("./types").LifecycleStatus,
  mergedWith?: string[],
): Promise<{ ok: boolean; status: string }> {
  const res = await fetch(`${BASE}/songs/${encodeURIComponent(songId)}/lifecycle`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, merged_with: mergedWith ?? [] }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Lifecycle update failed");
  }
  return res.json();
}

export async function fetchScrappedSongs(): Promise<import("./types").SongEntry[]> {
  const res = await fetch(`${BASE}/songs/scrapped`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch scrapped songs");
  return res.json();
}

export async function setUsesPartsFrom(
  songId: string,
  usesPartsFrom: string[],
): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE}/songs/${encodeURIComponent(songId)}/uses-parts-from`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ uses_parts_from: usesPartsFrom }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Uses-parts-from update failed");
  }
  return res.json();
}

export async function exportSample(segmentId: string): Promise<{ ok: boolean; dest: string }> {
  const res = await fetch(`${BASE}/samples/${encodeURIComponent(segmentId)}/export`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw Object.assign(
      new Error((err as { detail?: string }).detail ?? "Export failed"),
      { status: res.status }
    );
  }
  return res.json();
}

export async function fetchSides(): Promise<import("./types").SidesResponse> {
  const res = await fetch(`${BASE}/sides`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch sides");
  return res.json();
}

export async function assignSongToSide(
  side: string,
  songId: string,
  position: number,
): Promise<{ ok: boolean; side: string; songs: import("./types").SideSong[] }> {
  const res = await fetch(`${BASE}/sides/${encodeURIComponent(side)}/assign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ song_id: songId, position }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Assign failed");
  }
  return res.json();
}

export async function moveSongBetweenSides(
  fromSide: string,
  songId: string,
  toSide: string,
  toPosition: number,
): Promise<{ ok: boolean; side: string; songs: import("./types").SideSong[] }> {
  const res = await fetch(`${BASE}/sides/${encodeURIComponent(fromSide)}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ song_id: songId, to_side: toSide, to_position: toPosition }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Move failed");
  }
  return res.json();
}

export async function setLpConsideration(
  songId: string,
  status: import("./types").LpConsiderationStatus,
): Promise<{ ok: boolean; status: string }> {
  const res = await fetch(`${BASE}/songs/${encodeURIComponent(songId)}/lp-consideration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "LP consideration update failed");
  }
  return res.json();
}

export async function removeSongFromSide(
  side: string,
  songId: string,
): Promise<{ ok: boolean }> {
  const res = await fetch(
    `${BASE}/sides/${encodeURIComponent(side)}/songs/${encodeURIComponent(songId)}`,
    { method: "DELETE" },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Remove failed");
  }
  return res.json();
}
