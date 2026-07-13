"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchSongs, fetchActiveSong, fetchComposition, activateSong, advanceStage, addVersion,
  runNextPhase, getRunStatus, fetchLyrics, approveLyric, promotePhase,
  autoSplitMelody, assembleMelody, syncArrangement, fetchSongMixInfo, setMixFile, songMixStreamUrl, setBpm,
  fetchWorkOrders, fetchCollaborators, regressStage,
  setLifecycleStatus, fetchScrappedSongs, setUsesPartsFrom,
} from "@/lib/api";
import { Collaborator, CompositionEntry, LifecycleStatus, LyricCandidate, LyricsResponse, MIX_STAGES, MixStage, RegressionInfo, RunJob, SongEntry, WorkOrder } from "@/lib/types";
import WorkOrderHud from "@/components/WorkOrderHud";
import WorkOrderDrawer from "@/components/WorkOrderDrawer";
import { Combobox, ComboboxOption } from "@/components/Combobox";
import { NotesButton, SongNotesModal } from "@/components/SongNotes";

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

function RegressionModal({
  targetStage,
  info,
  onClose,
  onConfirm,
  confirming,
}: {
  targetStage: MixStage;
  info: RegressionInfo;
  onClose: () => void;
  onConfirm: (diaryEntry: string | null) => void;
  confirming: boolean;
}) {
  const [diaryEntry, setDiaryEntry] = useState("");

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-lg mx-4 flex flex-col shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 flex-shrink-0">
          <span className="text-sm font-semibold text-white font-sans">
            Move back to {STAGE_LABELS[targetStage]}?
          </span>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-200 text-lg leading-none transition-colors" aria-label="Close">×</button>
        </div>

        <div className="flex flex-col gap-3 px-5 py-4">
          {info.destructive && info.files_to_delete.length > 0 && (
            <div className="rounded border border-red-800 bg-red-900/20 px-3 py-2">
              <p className="text-xs font-sans text-red-300 mb-1.5 font-semibold">Files that will be deleted:</p>
              <ul className="max-h-32 overflow-y-auto space-y-0.5">
                {info.files_to_delete.map(f => (
                  <li key={f} className="text-[10px] font-mono text-red-400">{f}</li>
                ))}
              </ul>
            </div>
          )}

          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-sans text-zinc-500 uppercase tracking-wider">Reason (optional diary entry)</span>
            <textarea
              value={diaryEntry}
              onChange={e => setDiaryEntry(e.target.value)}
              rows={3}
              placeholder="e.g. Had instrumental melodies on vocal track — need clean lyrics pass"
              className="bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 resize-none transition-colors"
            />
          </label>

          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="flex-1 py-2 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => onConfirm(diaryEntry.trim() || null)}
              disabled={confirming}
              className={`flex-1 py-2 text-xs font-sans font-semibold rounded border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                info.destructive
                  ? "bg-red-800 border-red-700 text-red-100 hover:bg-red-700"
                  : "bg-zinc-700 border-zinc-600 text-zinc-100 hover:bg-zinc-600"
              }`}
            >
              {confirming ? "Moving…" : "Confirm"}
            </button>
          </div>
        </div>
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
  const [splitting, setSplitting] = useState(false);
  const [splitResult, setSplitResult] = useState<string | null>(null);
  const [assembling, setAssembling] = useState(false);
  const [assembleResult, setAssembleResult] = useState<string | null>(null);
  const [mixFile, setMixFileState] = useState<string | null>(null);
  const [mixPathInput, setMixPathInput] = useState("");
  const [settingMix, setSettingMix] = useState(false);
  const [showMixInput, setShowMixInput] = useState(false);
  const [conceptExpanded, setConceptExpanded] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncDone, setSyncDone] = useState(false);
  const [editingBpm, setEditingBpm] = useState(false);
  const [bpmInput, setBpmInput] = useState("");
  const [settingBpm, setSettingBpm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [workOrders, setWorkOrders] = useState<WorkOrder[]>([]);
  const [collaborators, setCollaborators] = useState<Collaborator[]>([]);
  const [workOrderDrawerOpen, setWorkOrderDrawerOpen] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [regressionModal, setRegressionModal] = useState<{ targetStage: MixStage; info: RegressionInfo } | null>(null);
  const [confirmingRegress, setConfirmingRegress] = useState(false);
  // Lifecycle panel state
  const [lifecyclePanelOpen, setLifecyclePanelOpen] = useState(false);
  const [abandonModal, setAbandonModal] = useState(false);
  const [scrapModal, setScrapModal] = useState(false);
  const [mergeModal, setMergeModal] = useState(false);
  const [mergeTarget, setMergeTarget] = useState<string>("");
  const [lifecyclePending, setLifecyclePending] = useState(false);
  const [lifecycleError, setLifecycleError] = useState<string | null>(null);
  const [usesPartsExpanded, setUsesPartsExpanded] = useState(false);
  const [scrappedSongs, setScrappedSongs] = useState<SongEntry[]>([]);
  const [selectedPartsFrom, setSelectedPartsFrom] = useState<string[]>([]);
  const [savingPartsFrom, setSavingPartsFrom] = useState(false);
  const lyricsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lifecycleScrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const updateLifecycleScrollState = useCallback(() => {
    const el = lifecycleScrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
  }, []);

  useEffect(() => {
    updateLifecycleScrollState();
    window.addEventListener("resize", updateLifecycleScrollState);
    return () => window.removeEventListener("resize", updateLifecycleScrollState);
  }, [updateLifecycleScrollState, composition]);

  const songOptions: ComboboxOption<string>[] = useMemo(() => {
    const sorted = [...songs].sort((a, b) => a.title.localeCompare(b.title) || a.thread_slug.localeCompare(b.thread_slug));
    return sorted.map(s => ({
      value: s.id,
      label: s.title,
      secondary: `${s.thread_slug} · ${s.id}`,
      keywords: [s.thread_slug, s.id],
    }));
  }, [songs]);

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
      if (active.active) {
        const mixInfo = await fetchSongMixInfo(active.active.id);
        setMixFileState(mixInfo.has_mix ? mixInfo.mix_file : null);
      } else {
        setMixFileState(null);
      }
      setLoadState("ready");
      // Load work orders + collaborators in the background (non-blocking)
      fetchWorkOrders().then(setWorkOrders).catch(() => {});
      fetchCollaborators().then(setCollaborators).catch(() => {});
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

  const handleRegressClick = async (targetStage: MixStage) => {
    setError(null);
    try {
      const result = await regressStage(targetStage, false, null) as RegressionInfo;
      setRegressionModal({ targetStage, info: result });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Regression check failed");
    }
  };

  const handleRegressConfirm = async (diaryEntry: string | null) => {
    if (!regressionModal) return;
    setConfirmingRegress(true);
    setError(null);
    try {
      await regressStage(regressionModal.targetStage, true, diaryEntry);
      setRegressionModal(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Regression failed");
    } finally {
      setConfirmingRegress(false);
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

  const handleSyncArrangement = async () => {
    setSyncing(true);
    setSyncDone(false);
    setError(null);
    try {
      await syncArrangement();
      setSyncDone(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Arrangement sync failed");
    } finally {
      setSyncing(false);
    }
  };

  const handleAutoSplit = async () => {
    setSplitting(true);
    setSplitResult(null);
    setError(null);
    try {
      const res = await autoSplitMelody();
      const results = res.results ?? [];
      const warnings = res.warnings ?? [];
      const splitResults = results.filter(r => !r.skipped);
      const skippedCount = results.length - splitResults.length;
      let message =
        splitResults.length > 0
          ? `Split ${splitResults.length} section(s): ${splitResults.map(r => r.section).join(", ")}`
          : "No sections were split";
      if (skippedCount > 0) {
        message += ` (${skippedCount} skipped)`;
      }
      if (warnings.length > 0) {
        message += ` — ⚠ ${warnings.join(" ")}`;
      }
      setSplitResult(message);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Auto-split failed");
    } finally {
      setSplitting(false);
    }
  };

  const handleSetMix = async () => {
    if (!mixPathInput.trim()) return;
    setSettingMix(true);
    setError(null);
    try {
      await setMixFile(mixPathInput.trim());
      setMixFileState(mixPathInput.trim());
      setMixPathInput("");
      setShowMixInput(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to set mix file");
    } finally {
      setSettingMix(false);
    }
  };

  const handleSetBpm = async () => {
    const bpm = parseInt(bpmInput, 10);
    if (isNaN(bpm) || bpm < 20 || bpm > 400) return;
    setSettingBpm(true);
    setError(null);
    try {
      await setBpm(bpm);
      await refresh();
      setEditingBpm(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "BPM update failed");
    } finally {
      setSettingBpm(false);
    }
  };

  const handleAssembleMelody = async () => {
    setAssembling(true);
    setAssembleResult(null);
    setError(null);
    try {
      const res = await assembleMelody();
      setAssembleResult(
        res.assembled_lyrics
          ? "assembled_melody.mid + assembled_lyrics.txt written"
          : "assembled_melody.mid written (no lyrics.txt found to assemble)",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Assemble failed");
    } finally {
      setAssembling(false);
    }
  };

  const handleLifecycle = async (status: LifecycleStatus, mergedWith?: string[]) => {
    if (!activeSong) return;
    setLifecyclePending(true);
    setLifecycleError(null);
    try {
      await setLifecycleStatus(activeSong.id, status, mergedWith);
      await refresh();
    } catch (e: unknown) {
      setLifecycleError((e as Error).message ?? "Failed to update lifecycle status");
    } finally {
      setAbandonModal(false);
      setScrapModal(false);
      setMergeModal(false);
      setLifecyclePending(false);
    }
  };

  const handleOpenUsesPartsFrom = async () => {
    setUsesPartsExpanded(true);
    try {
      const songs = await fetchScrappedSongs();
      setScrappedSongs(songs);
      setSelectedPartsFrom(activeSong?.uses_parts_from ?? []);  // uses_parts_from absent on legacy entries
    } catch {
      setScrappedSongs([]);
    }
  };

  const handleSavePartsFrom = async () => {
    if (!activeSong) return;
    setLifecycleError(null);
    setSavingPartsFrom(true);
    try {
      await setUsesPartsFrom(activeSong.id, selectedPartsFrom);
      await refresh();
    } catch {
      setLifecycleError("Failed to save — please try again");
    } finally {
      setSavingPartsFrom(false);
    }
  };

  const currentStageIdx = composition ? MIX_STAGES.indexOf(composition.current_stage) : -1;
  const promotedCandidate = lyricsData?.candidates.find(c => c.status === "promoted");

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 font-mono">
      {/* Diary / notes modal */}
      {notesOpen && activeSong && (
        <SongNotesModal song={activeSong} onClose={() => setNotesOpen(false)} onUpdated={refresh} />
      )}

      {/* Regression modal */}
      {regressionModal && (
        <RegressionModal
          targetStage={regressionModal.targetStage}
          info={regressionModal.info}
          onClose={() => setRegressionModal(null)}
          onConfirm={handleRegressConfirm}
          confirming={confirmingRegress}
        />
      )}

      {/* Lyric modal */}
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
        <Link href="/candidates" className="text-zinc-500 hover:text-zinc-300 text-xs font-sans transition-colors">
          candidate view
        </Link>
        <Link href="/sides" className="text-zinc-500 hover:text-zinc-300 text-xs font-sans transition-colors">
          sides
        </Link>
        <h1 className="text-lg font-bold text-white tracking-tight">Composition Board</h1>
        <div className="ml-auto flex items-center gap-3">
          {songOptions.length > 0 && (
            <Combobox
              options={songOptions}
              value={activeSong?.id ?? ""}
              onChange={async (id) => {
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
              triggerLabel={activeSong?.title ?? "Select song…"}
              placeholder="Search song title or thread…"
              className={loadState === "loading" ? "opacity-50 pointer-events-none" : ""}
            />
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
          {activeSong && (
            <NotesButton title={activeSong.title} onOpen={() => setNotesOpen(true)} label="Diary" />
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
          <Link href="/" className="text-blue-400 hover:text-blue-300 transition-colors">
            Go to Songs → Handoff to Logic
          </Link>
        </div>
      )}

      {loadState === "ready" && composition && (
        <div className="px-6 pt-4 pb-0">
          {activeSong?.concept && (
            <div className="mb-4 bg-zinc-900/60 border border-zinc-800 rounded p-3 font-sans text-sm text-zinc-400">
              <div className={conceptExpanded ? "" : "line-clamp-3"}>{activeSong.concept}</div>
              <button
                onClick={() => setConceptExpanded(e => !e)}
                className="mt-1 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
              >
                {conceptExpanded ? "Show less" : "Show more"}
              </button>
            </div>
          )}
          {/* BPM display + edit */}
          {activeSong?.bpm != null && (
            <div className="flex items-center gap-2 mb-3">
              {editingBpm ? (
                <>
                  <input
                    type="number"
                    value={bpmInput}
                    onChange={e => setBpmInput(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") handleSetBpm(); if (e.key === "Escape") setEditingBpm(false); }}
                    autoFocus
                    className="w-20 bg-zinc-900 border border-zinc-600 rounded px-2 py-0.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-blue-500"
                    min={20} max={400}
                  />
                  <button
                    onClick={handleSetBpm}
                    disabled={settingBpm}
                    className="px-2 py-0.5 text-[10px] font-sans rounded bg-blue-800 border border-blue-600 text-blue-200 hover:bg-blue-700 disabled:opacity-40 transition-colors"
                  >
                    {settingBpm ? "Saving…" : "Save"}
                  </button>
                  <button
                    onClick={() => setEditingBpm(false)}
                    className="text-[10px] font-sans text-zinc-500 hover:text-zinc-300 transition-colors"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <span className="text-[10px] font-mono text-zinc-500">{activeSong.bpm} BPM</span>
                  {activeSong.time_sig && (
                    <span className="text-[10px] font-mono text-zinc-600">· {activeSong.time_sig}</span>
                  )}
                  <button
                    onClick={() => { setBpmInput(String(activeSong.bpm)); setEditingBpm(true); }}
                    className="text-[10px] font-sans text-zinc-600 hover:text-zinc-400 transition-colors"
                  >
                    change
                  </button>
                </>
              )}
            </div>
          )}

          {mixFile && activeSong ? (
            <div className="flex items-center gap-3 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 mb-4">
              <span className="text-[10px] font-sans uppercase tracking-wide text-zinc-600 whitespace-nowrap" title="Full song mix, separate from any per-stage take shown below">
                Song mix
              </span>
              <audio
                key={mixFile}
                controls
                src={songMixStreamUrl(activeSong.id)}
                className="h-8 flex-1 min-w-0"
                style={{ colorScheme: "dark" }}
              />
              <button
                onClick={() => { setShowMixInput(v => !v); setMixPathInput(mixFile ?? ""); }}
                className="text-[10px] font-sans text-zinc-500 hover:text-zinc-300 whitespace-nowrap transition-colors"
              >
                change
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowMixInput(v => !v)}
              className="mb-4 text-[10px] font-sans text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              + attach mix file
            </button>
          )}
          {showMixInput && (
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                defaultValue={mixFile ?? "/Users/gabrielwalsh/Documents/Music Production/Earthly Frames/White/Listening/"}
                onChange={e => setMixPathInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleSetMix()}
                placeholder="/Users/gabrielwalsh/Documents/Music Production/Earthly Frames/White/Listening/"
                className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-xs font-mono text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
              />
              <button
                onClick={handleSetMix}
                disabled={settingMix || !mixPathInput.trim()}
                className="px-3 py-1.5 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {settingMix ? "Saving…" : "Set"}
              </button>
            </div>
          )}

          {/* ── Song Lifecycle ─────────────────────────────────── */}
          {activeSong && (
            <div className="mb-4 border border-zinc-800 rounded-lg overflow-hidden">
              <button
                onClick={() => setLifecyclePanelOpen(o => !o)}
                className="w-full flex items-center justify-between px-4 py-2.5 bg-zinc-900 text-zinc-400 text-sm font-sans hover:bg-zinc-800 transition-colors"
              >
                <span className="font-medium text-zinc-300">Song Lifecycle</span>
                <span className="text-xs text-zinc-600">{lifecyclePanelOpen ? "▲" : "▼"}</span>
              </button>

              {lifecyclePanelOpen && (
                <div className="px-4 py-3 bg-zinc-950 space-y-3">
                  {lifecycleError && (
                    <div className="text-xs font-sans text-red-400 bg-red-900/20 border border-red-800 rounded px-2 py-1">
                      {lifecycleError}
                    </div>
                  )}

                  {/* Terminal state banner */}
                  {activeSong.lifecycle_status ? (
                    <div className={`rounded px-3 py-2 text-sm font-sans border ${
                      activeSong.lifecycle_status === "merged"    ? "bg-indigo-900/30 border-indigo-800 text-indigo-300" :
                      activeSong.lifecycle_status === "abandoned" ? "bg-zinc-800/60 border-zinc-700 text-zinc-400" :
                                                                    "bg-amber-900/20 border-amber-800 text-amber-400"
                    }`}>
                      This song is marked as <strong>{activeSong.lifecycle_status}</strong>.
                      {activeSong.lifecycle_status === "merged" && (activeSong.merged_with?.length ?? 0) > 0 && (
                        <span className="block mt-0.5 text-xs opacity-70">
                          Merged with: {activeSong.merged_with.join(", ")}
                        </span>
                      )}
                    </div>
                  ) : (
                    /* Action buttons — only for active (non-terminal) songs */
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => setAbandonModal(true)}
                        className="px-3 py-1.5 text-xs font-sans rounded border border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200 transition-colors"
                      >
                        Abandon
                      </button>
                      <button
                        onClick={() => setScrapModal(true)}
                        className="px-3 py-1.5 text-xs font-sans rounded border border-amber-800 bg-amber-900/20 text-amber-400 hover:bg-amber-900/40 transition-colors"
                      >
                        Scrap
                      </button>
                      <button
                        onClick={() => { setMergeTarget(""); setMergeModal(true); }}
                        className="px-3 py-1.5 text-xs font-sans rounded border border-indigo-800 bg-indigo-900/20 text-indigo-400 hover:bg-indigo-900/40 transition-colors"
                      >
                        Merge into suite…
                      </button>
                    </div>
                  )}

                  {/* Uses parts from */}
                  <div className="border-t border-zinc-800 pt-3">
                    <button
                      onClick={() => usesPartsExpanded ? setUsesPartsExpanded(false) : handleOpenUsesPartsFrom()}
                      className="text-xs font-sans text-zinc-500 hover:text-zinc-300 transition-colors"
                    >
                      {usesPartsExpanded ? "▲ Uses parts from" : "▼ Uses parts from"}
                      {(activeSong.uses_parts_from?.length ?? 0) > 0 && (
                        <span className="ml-1.5 text-zinc-600">({activeSong.uses_parts_from.length})</span>
                      )}
                    </button>
                    {usesPartsExpanded && (
                      <div className="mt-2 space-y-1.5">
                        {scrappedSongs.length === 0 ? (
                          <p className="text-xs font-sans text-zinc-600">No scrapped songs available.</p>
                        ) : (
                          scrappedSongs.map(s => (
                            <label key={s.id} className="flex items-center gap-2 text-xs font-sans text-zinc-400 cursor-pointer hover:text-zinc-200">
                              <input
                                type="checkbox"
                                checked={selectedPartsFrom.includes(s.id)}
                                onChange={e => setSelectedPartsFrom(prev =>
                                  e.target.checked ? [...prev, s.id] : prev.filter(id => id !== s.id)
                                )}
                                className="accent-amber-500"
                              />
                              <span>{s.title}</span>
                              <span className="text-zinc-600 truncate">{s.id}</span>
                            </label>
                          ))
                        )}
                        {scrappedSongs.length > 0 && (
                          <button
                            onClick={handleSavePartsFrom}
                            disabled={savingPartsFrom}
                            className="mt-1 px-3 py-1 text-xs font-sans rounded border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 transition-colors"
                          >
                            {savingPartsFrom ? "Saving…" : "Save"}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Abandon confirmation ────────────────────────────── */}
          {abandonModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setAbandonModal(false)}>
              <div className="bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-sm mx-4 p-5 shadow-2xl" onClick={e => e.stopPropagation()}>
                <h2 className="text-base font-semibold text-white font-sans mb-2">Wait — don&apos;t give up!</h2>
                <p className="text-sm font-sans text-zinc-400 mb-4">
                  This song has been waiting patiently in the queue, dreaming of the day it becomes a banger.
                  Are you <em>absolutely sure</em> you want to abandon it to the void? It will be hidden from the main list but never forgotten.
                </p>
                <div className="flex justify-end gap-2">
                  <button onClick={() => setAbandonModal(false)} className="px-3 py-1.5 text-xs font-sans rounded border border-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
                    Save the song!
                  </button>
                  <button
                    onClick={() => handleLifecycle("abandoned")}
                    disabled={lifecyclePending}
                    className="px-3 py-1.5 text-xs font-sans rounded border border-zinc-600 bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40 transition-colors"
                  >
                    {lifecyclePending ? "Abandoning…" : "Yes, abandon it"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ── Scrap confirmation ──────────────────────────────── */}
          {scrapModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setScrapModal(false)}>
              <div className="bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-sm mx-4 p-5 shadow-2xl" onClick={e => e.stopPropagation()}>
                <h2 className="text-base font-semibold text-white font-sans mb-2">Scrap this song?</h2>
                <p className="text-sm font-sans text-zinc-400 mb-4">
                  Scrapping marks this song as a donor — its parts, ideas, and textures can still be referenced by other productions via &ldquo;Uses parts from.&rdquo;
                  It won&apos;t appear in the main list, but its contributions live on.
                </p>
                <div className="flex justify-end gap-2">
                  <button onClick={() => setScrapModal(false)} className="px-3 py-1.5 text-xs font-sans rounded border border-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
                    Cancel
                  </button>
                  <button
                    onClick={() => handleLifecycle("scrapped")}
                    disabled={lifecyclePending}
                    className="px-3 py-1.5 text-xs font-sans rounded border border-amber-700 bg-amber-900/30 text-amber-300 hover:bg-amber-900/50 disabled:opacity-40 transition-colors"
                  >
                    {lifecyclePending ? "Scrapping…" : "Scrap it"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ── Merge picker ────────────────────────────────────── */}
          {mergeModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setMergeModal(false)}>
              <div className="bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-sm mx-4 p-5 shadow-2xl" onClick={e => e.stopPropagation()}>
                <h2 className="text-base font-semibold text-white font-sans mb-1">Merge into suite</h2>
                <p className="text-sm font-sans text-zinc-400 mb-3">
                  Both songs will be marked <strong className="text-indigo-400">merged</strong> and linked to each other. Order doesn&apos;t matter — the suite binds them equally.
                </p>
                <select
                  aria-label="Select song to merge with"
                  value={mergeTarget}
                  onChange={e => setMergeTarget(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs font-sans text-zinc-200 mb-4 focus:outline-none focus:border-indigo-500"
                >
                  <option value="">— pick a song —</option>
                  {songs
                    .filter(s => s.id !== activeSong?.id && !["merged","abandoned","scrapped"].includes(s.stage))
                    .map(s => (
                      <option key={s.id} value={s.id}>{s.title}</option>
                    ))
                  }
                </select>
                <div className="flex justify-end gap-2">
                  <button onClick={() => setMergeModal(false)} className="px-3 py-1.5 text-xs font-sans rounded border border-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
                    Cancel
                  </button>
                  <button
                    onClick={() => mergeTarget && handleLifecycle("merged", [mergeTarget])}
                    disabled={lifecyclePending || !mergeTarget}
                    className="px-3 py-1.5 text-xs font-sans rounded border border-indigo-700 bg-indigo-900/30 text-indigo-300 hover:bg-indigo-900/50 disabled:opacity-40 transition-colors"
                  >
                    {lifecyclePending ? "Merging…" : "Merge"}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {loadState === "ready" && composition && (
        <div className="relative">
          {canScrollLeft && (
            <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-12 z-10 bg-gradient-to-r from-zinc-950 to-transparent" />
          )}
          <div
            ref={lifecycleScrollRef}
            onScroll={updateLifecycleScrollState}
            className="overflow-x-auto px-6 py-6"
          >
          <div className="flex gap-3 min-w-max">
            {MIX_STAGES.map((stage, idx) => {
              const isCurrent = stage === composition.current_stage;
              const isPast = idx < currentStageIdx;
              const isFuture = idx > currentStageIdx;
              const isLyrics = stage === "lyrics";
              const isVocalPlaceholders = stage === "vocal_placeholders";
              const isStructure = stage === "structure";
              const isRecording = stage === "recording";
              const isFinalMix = stage === "final_mix";

              return (
                <div
                  key={stage}
                  className={`flex flex-col w-52 rounded-lg border transition-colors ${
                    isFinalMix && (isCurrent || isPast)
                      ? "border-green-700 bg-zinc-900"
                      : isCurrent
                      ? "border-blue-600 bg-zinc-900"
                      : isPast
                      ? "border-zinc-700 bg-zinc-900/50"
                      : "border-zinc-800 bg-zinc-950"
                  }`}
                >
                  {/* Column header */}
                  <div className={`px-3 py-2.5 border-b ${
                    isFinalMix && (isCurrent || isPast) ? "border-green-700/50"
                    : isCurrent ? "border-blue-600/50"
                    : "border-zinc-800"
                  }`}>
                    <div className="flex items-center gap-2">
                      {isPast && !isFinalMix && (
                        <svg className="w-3 h-3 text-green-500 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                          <title>Phase reached — cards inside may still be in draft</title>
                          <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
                        </svg>
                      )}
                      {isFinalMix && (isCurrent || isPast) && (
                        <svg className="w-3 h-3 text-green-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                          <title>Phase reached — cards inside may still be in draft</title>
                          <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
                        </svg>
                      )}
                      {isCurrent && !isFinalMix && <span className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />}
                      <span className={`text-xs font-sans font-semibold truncate flex-1 ${
                        isFinalMix && (isCurrent || isPast) ? "text-green-400"
                        : isCurrent ? "text-blue-300"
                        : isPast ? "text-zinc-400"
                        : "text-zinc-600"
                      }`}>
                        {STAGE_LABELS[stage]}
                      </span>
                      {isCurrent && !isStructure && (
                        <button
                          onClick={() => {
                            const prevStage = MIX_STAGES[idx - 1];
                            if (prevStage) handleRegressClick(prevStage);
                          }}
                          title={`Move back to ${STAGE_LABELS[MIX_STAGES[idx - 1]]}`}
                          className="text-[10px] font-sans text-zinc-600 hover:text-zinc-300 transition-colors flex-shrink-0"
                          aria-label="Regress to previous stage"
                        >
                          ←
                        </button>
                      )}
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
                          <div className="flex items-center justify-between gap-1">
                            <span className={`text-xs font-semibold ${isCurrent ? "text-zinc-300" : "text-zinc-500"}`}>
                              v{v.version}
                            </span>
                            <span className="text-[10px] text-zinc-600 font-sans">
                              {new Date(v.created).toLocaleDateString()}
                            </span>
                          </div>
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

                  {/* Work order HUD — recording stage */}
                  {isRecording && (isCurrent || isPast) && (
                    <WorkOrderHud
                      workOrder={workOrders[0] ?? null}
                      collaborator={
                        workOrders[0]
                          ? collaborators.find(c => c.id === workOrders[0].collaborator_id) ?? null
                          : null
                      }
                      onOpen={() => setWorkOrderDrawerOpen(true)}
                    />
                  )}

                  {/* Sync arrangement.txt from Logic — structure stage */}
                  {isCurrent && isStructure && (
                    <div className="px-3 pb-1">
                      <button
                        onClick={handleSyncArrangement}
                        disabled={syncing}
                        title="Pull arrangement.txt from Logic project back into the production dir"
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {syncing ? "Syncing…" : syncDone ? "✓ Arrangement synced" : "Sync arrangement from Logic"}
                      </button>
                    </div>
                  )}

                  {/* Auto-split + assemble melody — vocal placeholders stage or approaching it */}
                  {isVocalPlaceholders && (isCurrent || (isFuture && idx === currentStageIdx + 1)) && (
                    <div className="px-3 pb-1 flex flex-col gap-1.5">
                      <button
                        onClick={handleAutoSplit}
                        disabled={splitting}
                        title="Split approved melody MIDIs by syllable count for ACE Studio import"
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {splitting ? "Splitting…" : "Auto-split for ACE"}
                      </button>
                      {splitResult && (
                        <p className="text-[10px] font-sans text-emerald-400 text-center">{splitResult}</p>
                      )}
                      <button
                        onClick={handleAssembleMelody}
                        disabled={assembling}
                        title="Assemble all split MIDIs into one full-length melody MIDI"
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {assembling ? "Assembling…" : "Assemble melody MIDI"}
                      </button>
                      {assembleResult && (
                        <p className="text-[10px] font-sans text-emerald-400 text-center">{assembleResult}</p>
                      )}
                    </div>
                  )}

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

                  {/* Advance button — not shown at final_mix (no next stage) */}
                  {isFuture && idx === currentStageIdx + 1 && !isFinalMix && (
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

                  {/* Completion panel — final mix only */}
                  {isFinalMix && (isCurrent || isPast) && (
                    <div className="px-3 pb-3 flex flex-col gap-2">
                      <div className="flex items-center gap-1.5 px-2 py-1.5 text-[10px] font-sans rounded border border-green-700/50 bg-green-900/20 text-green-400">
                        <svg className="w-3 h-3 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
                        </svg>
                        song complete
                      </div>
                      {isCurrent && (
                        <>
                          <button
                            onClick={() => {
                              if (composition?.logic_project_path) {
                                navigator.clipboard.writeText(composition.logic_project_path);
                              }
                            }}
                            disabled={!composition?.logic_project_path}
                            className="w-full py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                          >
                            copy project path
                          </button>
                          <button
                            onClick={() => window.open("https://music.apple.com/", "_blank")}
                            className="w-full py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-200 transition-colors"
                          >
                            open distributor
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          </div>
          {canScrollRight && (
            <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-12 z-10 bg-gradient-to-l from-zinc-950 to-transparent flex items-center justify-end pr-1">
              <span className="text-zinc-500 text-sm">›</span>
            </div>
          )}
        </div>
      )}

      {/* Work order drawer */}
      {workOrderDrawerOpen && (
        <WorkOrderDrawer
          workOrder={workOrders[0] ?? null}
          collaborator={
            workOrders[0]
              ? collaborators.find(c => c.id === workOrders[0].collaborator_id) ?? null
              : null
          }
          onClose={() => setWorkOrderDrawerOpen(false)}
          onSaved={saved => {
            setWorkOrders(prev => {
              const idx = prev.findIndex(w => w.collaborator_id === saved.collaborator_id);
              return idx >= 0
                ? prev.map((w, i) => (i === idx ? saved : w))
                : [...prev, saved];
            });
          }}
        />
      )}
    </div>
  );
}
