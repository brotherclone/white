"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { fetchCandidates, approveCandidate, rejectCandidate, setLabel, setUseCase, midiUrl, promotePhase, evolvePhase, fetchActiveSong, initSong, runNextPhase, runQuartetPhase, getRunStatus, fetchPipelineStatus, startHandoff, getHandoffStatus } from "@/lib/api";
import { Candidate, CandidateStatus, PipelineStatus, RunJob, SongEntry } from "@/lib/types";
import ScoreBar from "@/components/ScoreBar";
import ScorePanel from "@/components/ScorePanel";
import StatusBadge from "@/components/StatusBadge";
import MidiPlayer from "@/components/MidiPlayer";

const PHASES = ["chords", "drums", "bass", "melody", "quartet"];
const PIPELINE_PHASES = ["chords", "drums", "bass", "melody"];

type SortKey = "phase" | "section" | "id" | "template" | "composite_score" | "status" | "rank";

export default function CandidatesPage() {
  const router = useRouter();
  const [all, setAll] = useState<Candidate[]>([]);
  const [phaseFilter, setPhaseFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<"" | CandidateStatus>("");
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [focused, setFocused] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [promoting, setPromoting] = useState(false);
  const [evolving, setEvolving] = useState(false);
  const [running, setRunning] = useState(false);
  const [runningQuartet, setRunningQuartet] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [handoffing, setHandoffing] = useState(false);
  const [activeSong, setActiveSong] = useState<SongEntry | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const runPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const handoffPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refreshPipeline = useCallback(() => {
    fetchPipelineStatus().then(setPipeline).catch(() => {});
  }, []);

  useEffect(() => {
    fetchActiveSong().then(({ active }) => setActiveSong(active)).catch(() => {});
    refreshPipeline();
  }, [refreshPipeline]);

  useEffect(() => () => {
    if (runPollRef.current) clearInterval(runPollRef.current);
    if (handoffPollRef.current) clearInterval(handoffPollRef.current);
  }, []);

  const load = useCallback(async (phase?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchCandidates(phase || undefined);
      setAll(data);
    } catch (e: unknown) {
      const status = (e as { status?: number }).status;
      if (status === 503) {
        router.replace("/");
      } else {
        setError("Could not reach API — is the server running on localhost:8000?");
      }
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => { load(phaseFilter); }, [phaseFilter, load]);

  const visible = all
    .filter(c => !statusFilter || c.status === statusFilter)
    .sort((a, b) => {
      const av = (a as unknown as Record<string, unknown>)[sortKey] ?? "";
      const bv = (b as unknown as Record<string, unknown>)[sortKey] ?? "";
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortAsc ? cmp : -cmp;
    });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(s => !s);
    else { setSortKey(key); setSortAsc(true); }
  };

  const handleApprove = async (id: string) => {
    setAll(prev => prev.map(c => c.id === id ? { ...c, status: "approved" as CandidateStatus } : c));
    try { await approveCandidate(id); }
    catch { setAll(prev => prev.map(c => c.id === id ? { ...c, status: "pending" as CandidateStatus } : c)); }
  };

  const handleReject = async (id: string) => {
    setAll(prev => prev.map(c => c.id === id ? { ...c, status: "rejected" as CandidateStatus } : c));
    try { await rejectCandidate(id); }
    catch { setAll(prev => prev.map(c => c.id === id ? { ...c, status: "pending" as CandidateStatus } : c)); }
  };

  const handlePlay = (id: string) => setPlayingId(prev => prev === id ? null : id);

  const handleLabel = async (id: string, label: string) => {
    const prev_label = all.find(c => c.id === id)?.label;
    setAll(prev => prev.map(c => c.id === id ? { ...c, label } : c));
    try { await setLabel(id, label); }
    catch {
      setAll(prev => prev.map(c => c.id === id ? { ...c, label: prev_label ?? "" } : c));
      setError("Could not save label — edit reverted.");
    }
  };

  const handleUseCase = async (id: string, use_case: string) => {
    const prev_use_case = all.find(c => c.id === id)?.use_case;
    setAll(prev => prev.map(c => c.id === id ? { ...c, use_case } : c));
    try { await setUseCase(id, use_case); }
    catch {
      setAll(prev => prev.map(c => c.id === id ? { ...c, use_case: prev_use_case ?? "" } : c));
      setError("Could not save use case — edit reverted.");
    }
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (!focused || tag === "INPUT" || tag === "SELECT") return;

      const focusedCandidate = all.find(c => c.id === focused);
      if (!focusedCandidate) return;

      if (e.key === "a" && focusedCandidate.status === "pending") handleApprove(focused);
      if (e.key === "r" && focusedCandidate.status === "pending") handleReject(focused);
      if (e.key === "p") handlePlay(focused);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [all, focused]); // eslint-disable-line react-hooks/exhaustive-deps

  const showToast = (msg: string) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToast(msg);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  };

  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current); }, []);

  const handleInit = async () => {
    setError(null);
    setInitializing(true);
    try {
      await initSong();
      refreshPipeline();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Init failed");
    } finally {
      setInitializing(false);
    }
  };

  const handleRun = useCallback(async () => {
    setError(null);
    setRunning(true);
    try {
      await runNextPhase();
    } catch (e) {
      setRunning(false);
      setError(e instanceof Error ? e.message : "Run failed");
      return;
    }
    runPollRef.current = setInterval(async () => {
      try {
        const job = await getRunStatus();
        if (job.status === "done") {
          clearInterval(runPollRef.current!);
          runPollRef.current = null;
          setRunning(false);
          refreshPipeline();
          load(phaseFilter);
          showToast(`Phase "${job.phase}" complete`);
        } else if (job.status === "error") {
          clearInterval(runPollRef.current!);
          runPollRef.current = null;
          setRunning(false);
          setError(job.error ?? "Phase run failed");
          refreshPipeline();
        }
      } catch { /* transient */ }
    }, 3000);
  }, [phaseFilter, load, refreshPipeline]);

  const handleRunQuartet = async () => {
    setError(null);
    setRunningQuartet(true);
    try {
      await runQuartetPhase();
    } catch (e) {
      setRunningQuartet(false);
      setError(e instanceof Error ? e.message : "Strings run failed");
      return;
    }
    runPollRef.current = setInterval(async () => {
      try {
        const job = await getRunStatus();
        if (job.status === "done") {
          clearInterval(runPollRef.current!);
          runPollRef.current = null;
          setRunningQuartet(false);
          refreshPipeline();
          load(phaseFilter);
          showToast("Strings generation complete");
        } else if (job.status === "error") {
          clearInterval(runPollRef.current!);
          runPollRef.current = null;
          setRunningQuartet(false);
          setError(job.error ?? "Strings generation failed");
          refreshPipeline();
        }
      } catch { /* transient */ }
    }, 3000);
  };

  const handlePromote = async () => {
    if (!phaseFilter) return;
    setError(null);
    setPromoting(true);
    try {
      const res = await promotePhase(phaseFilter);
      showToast(`Promoted ${res.promoted_count} file(s) for ${phaseFilter}`);
      load(phaseFilter);
      refreshPipeline();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Promote failed");
    } finally {
      setPromoting(false);
    }
  };

  const handleEvolve = async () => {
    if (!phaseFilter) return;
    setError(null);
    setEvolving(true);
    try {
      const res = await evolvePhase(phaseFilter);
      showToast(`Added ${res.evolved_count} evolved candidate(s) for ${phaseFilter}`);
      load(phaseFilter);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Evolve failed");
    } finally {
      setEvolving(false);
    }
  };

  const handleHandoff = async () => {
    setHandoffing(true);
    setError(null);
    try {
      await startHandoff();
    } catch (e) {
      setHandoffing(false);
      setError(e instanceof Error ? e.message : "Handoff failed");
      return;
    }
    handoffPollRef.current = setInterval(async () => {
      try {
        const job: RunJob = await getHandoffStatus();
        if (job.status === "done") {
          clearInterval(handoffPollRef.current!);
          handoffPollRef.current = null;
          setHandoffing(false);
          showToast(`Handoff complete — Logic project ready`);
        } else if (job.status === "error") {
          clearInterval(handoffPollRef.current!);
          handoffPollRef.current = null;
          setHandoffing(false);
          setError(job.error ?? "Handoff failed");
        }
      } catch { /* transient */ }
    }, 2000);
  };

  const canPromote = !!phaseFilter && phaseFilter !== "all";
  const canEvolve = ["drums", "bass", "melody"].includes(phaseFilter);
  const needsInit = pipeline?.next_phase === "init_production";
  const canRun = !running && !needsInit && pipeline?.next_phase != null;
  const melodyPromoted = pipeline?.phases?.["melody"] === "promoted";
  const quartetStatus = pipeline?.phases?.["quartet"];
  const canRunQuartet = melodyPromoted && !quartetStatus && !running && !runningQuartet;

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k ? <span className="ml-1 text-zinc-400">{sortAsc ? "↑" : "↓"}</span> : null;

  const Th = ({ k, label, className = "" }: { k: SortKey; label: string; className?: string }) => (
    <th
      onClick={() => handleSort(k)}
      className={`px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-white ${className}`}
    >
      {label}<SortIcon k={k} />
    </th>
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 p-4 font-mono">
      {/* Breadcrumb */}
      {activeSong && (
        <div className="flex items-center gap-2 text-xs text-zinc-500 mb-3 font-sans">
          <Link href="/songs" className="hover:text-zinc-300 transition-colors">← Songs</Link>
          <span>/</span>
          <span className="text-zinc-300">{activeSong.title}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
        <h1 className="text-lg font-bold text-white tracking-tight">Candidate Browser</h1>
        <div className="flex gap-2 items-center flex-wrap">
          <select
            className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200 font-sans"
            value={phaseFilter}
            onChange={e => setPhaseFilter(e.target.value)}
          >
            <option value="">All phases</option>
            {PHASES.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
          <select
            className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200 font-sans"
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value as "" | CandidateStatus)}
          >
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
          <span className="text-zinc-500 text-sm">{visible.length} candidates</span>
          {/* Promote */}
          <button
            onClick={handlePromote}
            disabled={!canPromote || promoting}
            title={canPromote ? `Promote approved ${phaseFilter} candidates` : "Select a phase to enable promote"}
            className="px-3 py-1 text-sm rounded font-medium transition-colors bg-emerald-700 hover:bg-emerald-600 text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {promoting ? "Promoting…" : "Promote"}
          </button>
          {/* Evolve */}
          {canEvolve && (
            <button
              onClick={handleEvolve}
              disabled={evolving}
              title={`Breed evolved ${phaseFilter} candidates`}
              className="px-3 py-1 text-sm rounded font-medium transition-colors bg-violet-700 hover:bg-violet-600 text-white disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {evolving ? "Evolving…" : "Evolve"}
            </button>
          )}
        </div>
      </div>

      {/* Pipeline status */}
      {pipeline && (
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          {pipeline.phase_order.filter(p => PIPELINE_PHASES.includes(p)).map(phase => {
            const status = pipeline.phases[phase] ?? "pending";
            const isNext = pipeline.next_phase === phase;
            const color =
              status === "promoted" ? "text-emerald-400" :
              status === "generated" ? "text-blue-400" :
              status === "in_progress" ? "text-yellow-400" :
              isNext ? "text-zinc-300" : "text-zinc-600";
            return (
              <span key={phase} className={`text-xs font-sans ${color}`}>
                {status === "promoted" ? "✓" : status === "generated" ? "●" : status === "in_progress" ? "⟳" : isNext ? "→" : "·"} {phase}
              </span>
            );
          })}
          {/* Quartet (strings) — parallel phase, shown when melody is promoted */}
          {melodyPromoted && (
            quartetStatus === "promoted" ? (
              <span className="text-xs font-sans text-emerald-400">✓ strings</span>
            ) : quartetStatus === "generated" ? (
              <span className="text-xs font-sans text-blue-400">● strings</span>
            ) : quartetStatus === "in_progress" || runningQuartet ? (
              <span className="text-xs font-sans text-yellow-400">⟳ strings…</span>
            ) : canRunQuartet ? (
              <button
                onClick={handleRunQuartet}
                className="px-3 py-1 text-xs rounded font-medium transition-colors bg-zinc-700 hover:bg-zinc-600 text-zinc-200"
              >
                Run Strings
              </button>
            ) : null
          )}
          {needsInit && (
            <button
              onClick={handleInit}
              disabled={initializing}
              className="ml-auto px-3 py-1 text-xs rounded font-medium transition-colors bg-zinc-600 hover:bg-zinc-500 text-white disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {initializing ? "Initializing…" : "Initialize"}
            </button>
          )}
          {canRun && pipeline.next_phase !== "lyrics" && (
            <button
              onClick={handleRun}
              disabled={running}
              className="ml-auto px-3 py-1 text-xs rounded font-medium transition-colors bg-blue-700 hover:bg-blue-600 text-white disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {running ? `Running ${pipeline.next_phase}…` : `Run ${pipeline.next_phase}`}
            </button>
          )}
          {canRun && pipeline.next_phase === "lyrics" && (
            <Link
              href="/board"
              className="ml-auto px-3 py-1 text-xs rounded font-medium transition-colors bg-violet-700 hover:bg-violet-600 text-white"
            >
              Composition Board →
            </Link>
          )}
          {running && !canRun && (
            <span className="ml-auto text-xs text-yellow-400 font-sans">Running {pipeline.next_phase}…</span>
          )}
          {melodyPromoted && (
            <button
              onClick={handleHandoff}
              disabled={handoffing || running}
              className="ml-auto flex items-center gap-1.5 px-3 py-1 text-xs rounded font-medium transition-colors bg-indigo-700 hover:bg-indigo-600 text-white disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {handoffing ? (
                <>
                  <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Handing off…
                </>
              ) : "Handoff to Logic"}
            </button>
          )}
        </div>
      )}

      {toast && (
        <div className="bg-emerald-900/50 border border-emerald-700 rounded p-3 mb-4 text-emerald-200 text-sm font-sans">{toast}</div>
      )}

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded p-3 mb-4 text-red-300 text-sm font-sans">{error}</div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead className="bg-zinc-900 text-zinc-400 border-b border-zinc-800">
            <tr>
              <Th k="phase" label="Phase" className="w-24" />
              <Th k="section" label="Section" className="w-28" />
              <Th k="id" label="ID" className="min-w-44" />
              <Th k="template" label="Template" className="min-w-36" />
              <Th k="composite_score" label="Score" className="min-w-44" />
              <Th k="status" label="Status" className="w-28" />
              <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider w-32">Label</th>
              <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider w-28">Use</th>
              <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider w-28">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={9} className="px-3 py-10 text-center text-zinc-500 font-sans">Loading…</td></tr>
            )}
            {!loading && visible.length === 0 && (
              <tr><td colSpan={9} className="px-3 py-10 text-center text-zinc-500 font-sans">No candidates found.</td></tr>
            )}
            {visible.map(c => (
              <React.Fragment key={c.id}>
                <tr
                  tabIndex={0}
                  onClick={() => { setExpanded(e => e === c.id ? null : c.id); setFocused(c.id); }}
                  onFocus={() => setFocused(c.id)}
                  className={[
                    "border-b border-zinc-800/60 cursor-pointer transition-colors outline-none",
                    focused === c.id ? "bg-zinc-800/70 ring-1 ring-inset ring-blue-600" : "hover:bg-zinc-900",
                    c.status === "approved" || c.status === "accepted" ? "bg-green-950/20" : "",
                    c.status === "rejected" ? "bg-red-950/20 opacity-60" : "",
                  ].join(" ")}
                >
                  <td className="px-3 py-2 text-zinc-400">{c.phase}</td>
                  <td className="px-3 py-2 text-zinc-400">{c.section || <span className="text-zinc-600">—</span>}</td>
                  <td className="px-3 py-2 text-zinc-200">{c.id}</td>
                  <td className="px-3 py-2 text-zinc-300 max-w-xs truncate font-sans text-xs" title={c.template}>{c.template}</td>
                  <td className="px-3 py-2"><ScoreBar value={c.composite_score} /></td>
                  <td className="px-3 py-2"><StatusBadge status={c.status} /></td>
                  <td className="px-3 py-2" onClick={e => e.stopPropagation()}>
                    <input
                      type="text"
                      defaultValue={c.label ?? ""}
                      placeholder="unlabeled"
                      onBlur={e => handleLabel(c.id, e.target.value)}
                      onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                      className="w-full bg-transparent border border-zinc-700 rounded px-1.5 py-0.5 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-blue-500"
                    />
                  </td>
                  <td className="px-3 py-2" onClick={e => e.stopPropagation()}>
                    {c.use_case ? (
                      <button
                        onClick={() => handleUseCase(c.id, c.use_case === "vocal" ? "instrumental" : "vocal")}
                        className={`px-2 py-0.5 text-xs rounded font-medium transition-colors ${
                          c.use_case === "vocal"
                            ? "bg-violet-800 hover:bg-violet-700 text-violet-100"
                            : "bg-zinc-700 hover:bg-zinc-600 text-zinc-200"
                        }`}
                      >
                        {c.use_case === "vocal" ? "vocal" : "instr"}
                      </button>
                    ) : <span className="text-zinc-600">—</span>}
                  </td>
                  <td className="px-3 py-2" onClick={e => e.stopPropagation()}>
                    <div className="flex gap-1">
                      <button
                        onClick={() => handleApprove(c.id)}
                        disabled={c.status === "approved" || c.status === "accepted"}
                        title="Approve (a)"
                        className="px-2 py-0.5 text-xs rounded bg-green-800 hover:bg-green-700 text-green-100 font-medium disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                      >✓</button>
                      <button
                        onClick={() => handleReject(c.id)}
                        disabled={c.status === "approved" || c.status === "accepted" || c.status === "rejected"}
                        title="Reject (r)"
                        className="px-2 py-0.5 text-xs rounded bg-red-900 hover:bg-red-800 text-red-100 font-medium disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                      >✗</button>
                      <button
                        onClick={() => handlePlay(c.id)}
                        title="Play (p)"
                        className={`px-2 py-0.5 text-xs rounded font-medium transition-colors ${
                          playingId === c.id ? "bg-blue-700 text-white" : "bg-zinc-700 hover:bg-zinc-600 text-zinc-200"
                        }`}
                      >▶</button>
                    </div>
                  </td>
                </tr>
                {playingId === c.id && (
                  <tr key={`${c.id}-player`} className="bg-zinc-900/60 border-b border-zinc-800/60">
                    <td colSpan={9} className="px-4 py-3">
                      <MidiPlayer url={midiUrl(c.id)} />
                    </td>
                  </tr>
                )}
                {expanded === c.id && (
                  <tr className="bg-zinc-900/30 border-b border-zinc-800/60">
                    <td colSpan={9} className="px-4 py-3">
                      <ScorePanel scores={c.scores} />
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 text-xs text-zinc-600 flex gap-4 font-sans flex-wrap">
        <span><kbd className="bg-zinc-800 px-1 rounded font-mono">a</kbd> approve focused row</span>
        <span><kbd className="bg-zinc-800 px-1 rounded font-mono">r</kbd> reject</span>
        <span><kbd className="bg-zinc-800 px-1 rounded font-mono">p</kbd> play / stop</span>
        <span>click row to expand score detail</span>
      </div>
    </div>
  );
}
