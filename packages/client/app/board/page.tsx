"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchSongs, fetchActiveSong, fetchComposition, activateSong, advanceStage, addVersion,
  updateVersionNotes, runNextPhase, getRunStatus, fetchLyrics, approveLyric, promotePhase,
} from "@/lib/api";
import { CompositionEntry, LyricCandidate, LyricsResponse, MIX_STAGES, MixStage, RunJob, SongEntry } from "@/lib/types";

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

const VERDICT_COLORS: Record<string, string> = {
  "splits needed":      "text-red-400 bg-red-900/30 border-red-800",
  "tight but workable": "text-yellow-400 bg-yellow-900/30 border-yellow-800",
  "paste-ready":        "text-green-400 bg-green-900/30 border-green-800",
  "spacious":           "text-blue-400 bg-blue-900/30 border-blue-800",
};

function LyricModal({
  candidate,
  readOnly,
  onClose,
  onPromote,
  promoting,
}: {
  candidate: LyricCandidate;
  readOnly: boolean;
  onClose: () => void;
  onPromote: () => void;
  promoting: boolean;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-lg mx-4 flex flex-col max-h-[80vh] shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 flex-shrink-0">
          <span className="text-sm font-semibold text-white font-sans">
            {readOnly ? "Lyrics — promoted" : `Lyrics — v${candidate.rank}`}
          </span>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 text-lg leading-none transition-colors"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Metadata */}
        <div className="flex items-center gap-2 px-5 py-2.5 border-b border-zinc-800/60 flex-shrink-0">
          {candidate.match != null && (
            <span className="text-[10px] font-sans text-zinc-400">
              match <span className="text-zinc-200 font-semibold">{(candidate.match * 100).toFixed(0)}%</span>
            </span>
          )}
          {candidate.fitting_verdict && (
            <span className={`text-[10px] font-sans px-1.5 py-0.5 rounded border ${VERDICT_COLORS[candidate.fitting_verdict] ?? "text-zinc-400 bg-zinc-800 border-zinc-700"}`}>
              {candidate.fitting_verdict}
            </span>
          )}
        </div>

        {/* Lyric text */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap leading-relaxed">
            {candidate.text}
          </pre>
        </div>

        {/* Footer */}
        {!readOnly && (
          <div className="px-5 py-3.5 border-t border-zinc-800 flex-shrink-0">
            <button
              onClick={onPromote}
              disabled={promoting}
              className="w-full py-2 text-sm font-sans font-semibold rounded-lg bg-violet-700 hover:bg-violet-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {promoting ? "Promoting…" : "Promote"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

type LoadState = "loading" | "not_initialized" | "ready" | "error";

export default function BoardPage() {
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [activeSong, setActiveSong] = useState<SongEntry | null>(null);
  const [composition, setComposition] = useState<CompositionEntry | null>(null);
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [lyricsData, setLyricsData] = useState<LyricsResponse | null>(null);
  const [modal, setModal] = useState<{ candidate: LyricCandidate; readOnly: boolean } | null>(null);
  const [promoting, setPromoting] = useState(false);
  const [advancingTo, setAdvancingTo] = useState<MixStage | null>(null);
  const [addingVersion, setAddingVersion] = useState(false);
  const [generatingLyrics, setGeneratingLyrics] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const notesSaveTimers = useRef<Record<number, ReturnType<typeof setTimeout>>>({});
  const lyricsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refreshLyrics = useCallback(async () => {
    try {
      const data = await fetchLyrics();
      setLyricsData(data);
    } catch (e: unknown) {
      if ((e as { status?: number }).status === 404) setLyricsData(null);
      // 503 = no active song; other errors silently ignored
    }
  }, []);

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
    fetchSongs().then(setSongs).catch(() => {});
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (loadState === "ready") refreshLyrics();
  }, [loadState, refreshLyrics]);

  useEffect(() => () => {
    if (lyricsPollRef.current) clearInterval(lyricsPollRef.current);
  }, []);

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

  const handleGenerateLyrics = async () => {
    setGeneratingLyrics(true);
    setError(null);
    try {
      await runNextPhase();
    } catch (e) {
      setGeneratingLyrics(false);
      setError(e instanceof Error ? e.message : "Lyrics generation failed");
      return;
    }
    lyricsPollRef.current = setInterval(async () => {
      try {
        const job: RunJob = await getRunStatus();
        if (job.status === "done") {
          clearInterval(lyricsPollRef.current!);
          lyricsPollRef.current = null;
          setGeneratingLyrics(false);
          refreshLyrics();
        } else if (job.status === "error") {
          clearInterval(lyricsPollRef.current!);
          lyricsPollRef.current = null;
          setGeneratingLyrics(false);
          setError(job.error ?? "Lyrics generation failed");
        }
      } catch { /* transient */ }
    }, 3000);
  };

  const handlePromoteLyric = async () => {
    if (!modal) return;
    setPromoting(true);
    setError(null);
    try {
      await approveLyric(modal.candidate.id);
      await promotePhase("lyrics");
      setModal(null);
      await refreshLyrics();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Promotion failed");
    } finally {
      setPromoting(false);
    }
  };

  const handleNotesChange = (version: number, notes: string) => {
    if (notesSaveTimers.current[version]) clearTimeout(notesSaveTimers.current[version]);
    notesSaveTimers.current[version] = setTimeout(() => {
      updateVersionNotes(version, notes).catch(() => {});
    }, 600);
  };

  const currentStageIdx = composition ? MIX_STAGES.indexOf(composition.current_stage) : -1;
  const promotedCandidate = lyricsData?.candidates.find(c => c.status === "promoted");

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 font-mono">
      {/* Modal */}
      {modal && (
        <LyricModal
          candidate={modal.candidate}
          readOnly={modal.readOnly}
          onClose={() => setModal(null)}
          onPromote={handlePromoteLyric}
          promoting={promoting}
        />
      )}

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
              className="text-xs font-sans bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
              value={activeSong?.id ?? ""}
              onChange={async (e) => {
                const id = e.target.value;
                if (!id || id === activeSong?.id) return;
                setLoadState("loading");
                try {
                  await activateSong(id);
                  await refresh();
                  await refreshLyrics();
                } catch {
                  setLoadState("error");
                }
              }}
              disabled={loadState === "loading"}
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

      {loadState === "ready" && composition && (
        <div className="overflow-x-auto px-6 py-6">
          <div className="flex gap-3 min-w-max">
            {MIX_STAGES.map((stage, idx) => {
              const isCurrent = stage === composition.current_stage;
              const isPast = idx < currentStageIdx;
              const isFuture = idx > currentStageIdx;
              const isLyrics = stage === "lyrics";

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
                      {isCurrent && <span className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />}
                      <span className={`text-xs font-sans font-semibold truncate ${
                        isCurrent ? "text-blue-300" : isPast ? "text-zinc-400" : "text-zinc-600"
                      }`}>
                        {STAGE_LABELS[stage]}
                      </span>
                    </div>
                  </div>

                  {/* Column body */}
                  <div className="flex-1 px-3 py-3 flex flex-col gap-2 min-h-32">
                    {/* Version cards */}
                    {(isCurrent || isPast) && composition.versions
                      .filter(v => v.stage === stage)
                      .map(v => (
                        <div
                          key={v.version}
                          className={`rounded border px-2.5 py-2 ${
                            isCurrent ? "bg-zinc-800 border-zinc-700" : "bg-zinc-900 border-zinc-800 opacity-70"
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

                    {/* Lyrics version buttons */}
                    {isCurrent && isLyrics && lyricsData?.status === "pending" && (
                      <div className="flex gap-1 flex-wrap">
                        {lyricsData.candidates.map(c => (
                          <button
                            key={c.id}
                            onClick={() => setModal({ candidate: c, readOnly: false })}
                            disabled={modal !== null}
                            className="px-2 py-1 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-violet-900 hover:border-violet-700 hover:text-violet-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                          >
                            v{c.rank}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* See Lyrics button after promotion */}
                    {isLyrics && lyricsData?.status === "promoted" && promotedCandidate && (
                      <button
                        onClick={() => setModal({ candidate: promotedCandidate, readOnly: true })}
                        className="w-full py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
                      >
                        See Lyrics
                      </button>
                    )}
                  </div>

                  {/* Generate Lyrics button — only when no candidates yet */}
                  {isCurrent && isLyrics && !lyricsData && (
                    <div className="px-3 pb-3">
                      <button
                        onClick={handleGenerateLyrics}
                        disabled={generatingLyrics}
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 text-[10px] font-sans rounded bg-violet-900 border border-violet-700 text-violet-200 hover:bg-violet-800 hover:border-violet-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {generatingLyrics ? (
                          <>
                            <svg className="w-2.5 h-2.5 animate-spin" viewBox="0 0 24 24" fill="none">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                            </svg>
                            Generating…
                          </>
                        ) : "Generate Lyrics"}
                      </button>
                    </div>
                  )}

                  {/* Advance button */}
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
