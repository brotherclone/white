"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { fetchSongs, fetchActiveSong, fetchComposition, advanceStage, addVersion, updateVersionNotes } from "@/lib/api";
import { CompositionEntry, MIX_STAGES, MixStage, SongEntry } from "@/lib/types";

const STAGE_LABELS: Record<MixStage, string> = {
  structure:          "Structure",
  lyrics:             "Lyrics",
  recording:          "Recording",
  vocal_placeholders: "Vocal Placeholders",
  augmentation:       "Augmentation",
  cleaning:           "Cleaning",
  rough_mix:          "Rough Mix",
  mix_candidate:      "Mix Candidate",
  final_mix:          "Final Mix",
};

type LoadState = "loading" | "no_active" | "not_initialized" | "ready" | "error";

export default function BoardPage() {
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [activeSong, setActiveSong] = useState<SongEntry | null>(null);
  const [composition, setComposition] = useState<CompositionEntry | null>(null);
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [advancingTo, setAdvancingTo] = useState<MixStage | null>(null);
  const [addingVersion, setAddingVersion] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const notesSaveTimers = useRef<Record<number, ReturnType<typeof setTimeout>>>({});

  const refresh = useCallback(async () => {
    try {
      const [comp, active] = await Promise.all([fetchComposition(), fetchActiveSong()]);
      if ("status" in comp && comp.status === "not_initialized") {
        setLoadState("not_initialized");
        return;
      }
      setComposition(comp as CompositionEntry);
      setActiveSong(active.active);
      setLoadState("ready");
    } catch {
      setLoadState("error");
    }
  }, []);

  useEffect(() => {
    fetchSongs()
      .then(setSongs)
      .catch(() => {});
    refresh();
  }, [refresh]);

  const handleAdvance = async (stage: MixStage) => {
    setAdvancingTo(stage);
    setError(null);
    try {
      await advanceStage(stage);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stage update failed");
    } finally {
      setAdvancingTo(null);
    }
  };

  const handleAddVersion = async () => {
    setAddingVersion(true);
    setError(null);
    try {
      await addVersion();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Add version failed");
    } finally {
      setAddingVersion(false);
    }
  };

  const handleNotesChange = (version: number, notes: string) => {
    if (notesSaveTimers.current[version]) clearTimeout(notesSaveTimers.current[version]);
    notesSaveTimers.current[version] = setTimeout(() => {
      updateVersionNotes(version, notes).catch(() => {});
    }, 600);
  };

  const currentStageIdx = composition
    ? MIX_STAGES.indexOf(composition.current_stage)
    : -1;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 font-mono">
      {/* Header */}
      <div className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4">
        <Link href="/" className="text-zinc-500 hover:text-zinc-300 text-xs font-sans transition-colors">
          ← home
        </Link>
        <h1 className="text-lg font-bold text-white tracking-tight">Composition Board</h1>
        {activeSong && (
          <span className="text-zinc-500 text-xs font-sans ml-2 truncate">{activeSong.title}</span>
        )}
        <div className="ml-auto flex items-center gap-3">
          {songs.length > 0 && (
            <select
              className="text-xs font-sans bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-600"
              value={activeSong?.id ?? ""}
              onChange={() => {}}
            >
              {songs.map(s => (
                <option key={s.id} value={s.id}>{s.title}</option>
              ))}
            </select>
          )}
          {composition && (
            <button
              onClick={handleAddVersion}
              disabled={addingVersion}
              className="px-3 py-1.5 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {addingVersion ? "Adding…" : `+ Version (v${composition.current_version})`}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mx-6 mt-4 bg-red-900/40 border border-red-700 rounded p-3 text-red-300 text-sm font-sans">
          {error}
        </div>
      )}

      {/* Loading / empty states */}
      {loadState === "loading" && (
        <div className="flex items-center justify-center h-64 text-zinc-500 text-sm font-sans">Loading…</div>
      )}
      {loadState === "error" && (
        <div className="flex items-center justify-center h-64 text-zinc-500 text-sm font-sans">
          Could not reach API — is the server running?
        </div>
      )}
      {loadState === "not_initialized" && (
        <div className="flex flex-col items-center justify-center h-64 gap-3 text-zinc-500 text-sm font-sans">
          <span>No composition initialized.</span>
          <Link href="/songs" className="text-blue-400 hover:text-blue-300 transition-colors">
            Go to Songs → Handoff to Logic
          </Link>
        </div>
      )}

      {/* Board */}
      {loadState === "ready" && composition && (
        <div className="overflow-x-auto px-6 py-6">
          <div className="flex gap-3 min-w-max">
            {MIX_STAGES.map((stage, idx) => {
              const isCurrent = stage === composition.current_stage;
              const isPast = idx < currentStageIdx;
              const isFuture = idx > currentStageIdx;

              return (
                <div
                  key={stage}
                  className={`flex flex-col w-52 rounded-lg border transition-colors ${
                    isCurrent
                      ? "border-blue-600 bg-zinc-900"
                      : isPast
                      ? "border-zinc-700 bg-zinc-900/50"
                      : "border-zinc-800 bg-zinc-950"
                  }`}
                >
                  {/* Column header */}
                  <div className={`px-3 py-2.5 border-b ${isCurrent ? "border-blue-600/50" : "border-zinc-800"}`}>
                    <div className="flex items-center gap-2">
                      {isPast && (
                        <svg className="w-3 h-3 text-green-500 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
                        </svg>
                      )}
                      {isCurrent && (
                        <span className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />
                      )}
                      <span className={`text-xs font-sans font-semibold truncate ${
                        isCurrent ? "text-blue-300" : isPast ? "text-zinc-400" : "text-zinc-600"
                      }`}>
                        {STAGE_LABELS[stage]}
                      </span>
                    </div>
                  </div>

                  {/* Column body */}
                  <div className="flex-1 px-3 py-3 flex flex-col gap-2 min-h-32">
                    {/* Version cards — editable notes on all versions */}
                    {(isCurrent || isPast) && composition.versions
                      .filter(v => v.stage === stage)
                      .map(v => (
                        <div
                          key={v.version}
                          className={`rounded border px-2.5 py-2 ${
                            isCurrent
                              ? "bg-zinc-800 border-zinc-700"
                              : "bg-zinc-900 border-zinc-800 opacity-70"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-1 mb-1.5">
                            <span className={`text-xs font-semibold ${isCurrent ? "text-zinc-300" : "text-zinc-500"}`}>
                              v{v.version}
                            </span>
                            <span className="text-[10px] text-zinc-600 font-sans">
                              {new Date(v.created).toLocaleDateString()}
                            </span>
                          </div>
                          <input
                            type="text"
                            defaultValue={v.notes ?? ""}
                            placeholder="add note…"
                            onChange={e => handleNotesChange(v.version, e.target.value)}
                            className="w-full bg-transparent text-[10px] font-sans text-zinc-400 placeholder-zinc-600 focus:outline-none focus:text-zinc-200 transition-colors"
                          />
                        </div>
                      ))
                    }
                  </div>

                  {/* Advance button — only show on the stage immediately after current */}
                  {isFuture && idx === currentStageIdx + 1 && (
                    <div className="px-3 pb-3">
                      <button
                        onClick={() => handleAdvance(stage)}
                        disabled={advancingTo !== null}
                        className="w-full py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {advancingTo === stage ? "Moving…" : `Move to ${STAGE_LABELS[stage]}`}
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Version history */}
          {composition.versions.length > 0 && (
            <div className="mt-8 border-t border-zinc-800 pt-6">
              <h2 className="text-xs font-sans font-semibold text-zinc-500 uppercase tracking-wider mb-3">
                Version History
              </h2>
              <div className="flex flex-col gap-1.5">
                {[...composition.versions].reverse().map(v => (
                  <div key={v.version} className="flex items-baseline gap-3 text-xs font-sans">
                    <span className="text-zinc-400 w-8 flex-shrink-0">v{v.version}</span>
                    <span className="text-zinc-600 w-28 flex-shrink-0">{new Date(v.created).toLocaleString()}</span>
                    <span className="text-zinc-500 w-36 flex-shrink-0">{STAGE_LABELS[v.stage as MixStage] ?? v.stage}</span>
                    <input
                      type="text"
                      defaultValue={v.notes ?? ""}
                      placeholder="—"
                      onChange={e => handleNotesChange(v.version, e.target.value)}
                      className="flex-1 bg-transparent text-zinc-400 placeholder-zinc-700 focus:outline-none focus:text-zinc-200 transition-colors"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
