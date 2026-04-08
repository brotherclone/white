"use client";

import React, { useEffect, useState, useCallback } from "react";
import { fetchCandidates, approveCandidate, rejectCandidate, setLabel, setUseCase, midiUrl } from "@/lib/api";
import { Candidate, CandidateStatus } from "@/lib/types";
import ScoreBar from "@/components/ScoreBar";
import ScorePanel from "@/components/ScorePanel";
import StatusBadge from "@/components/StatusBadge";
import MidiPlayer from "@/components/MidiPlayer";

const PHASES = ["chords", "drums", "bass", "melody", "quartet"];

type SortKey = "phase" | "section" | "id" | "template" | "composite_score" | "status" | "rank";

export default function Home() {
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

  const load = useCallback(async (phase?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchCandidates(phase || undefined);
      setAll(data);
    } catch {
      setError("Could not reach API — is the server running on localhost:8000?");
    } finally {
      setLoading(false);
    }
  }, []);

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
    setAll(prev => prev.map(c => c.id === id ? { ...c, label } : c));
    try { await setLabel(id, label); }
    catch { /* silent — local state already updated optimistically */ }
  };

  const handleUseCase = async (id: string, use_case: string) => {
    setAll(prev => prev.map(c => c.id === id ? { ...c, use_case } : c));
    try { await setUseCase(id, use_case); }
    catch { /* silent */ }
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
        </div>
      </div>

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
                      value={c.label ?? ""}
                      placeholder="unlabeled"
                      onChange={e => handleLabel(c.id, e.target.value)}
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
