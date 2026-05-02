"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchSongs, activateSong, initSong, startGenerate, getGenerateStatus, startHandoff, getHandoffStatus } from "@/lib/api";
import { RunJob, SongEntry } from "@/lib/types";

const COLOR_MAP: Record<string, string> = {
  red: "#dc2626",    r: "#dc2626",
  orange: "#ea580c", o: "#ea580c",
  yellow: "#ca8a04", y: "#ca8a04",
  green: "#16a34a",  g: "#16a34a",
  blue: "#2563eb",   b: "#2563eb",
  indigo: "#4f46e5", i: "#4f46e5",
  violet: "#7c3aed", v: "#7c3aed",
  coral: "#f97316",  c: "#f97316",
  black: "#3f3f46",  k: "#3f3f46",
  white: "#a1a1aa",  w: "#a1a1aa",
};

function colorDot(name: string | null) {
  const key = (name ?? "").toLowerCase();
  const bg = COLOR_MAP[key] ?? "#71717a";
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
      style={{ backgroundColor: bg }}
      title={name ?? "unknown"}
    />
  );
}

type Toast = { kind: "success" | "error"; message: string };

export default function SongIndexPage() {
  const router = useRouter();
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [generating, setGenerating] = useState(false);
  const [handoffingId, setHandoffingId] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const handoffPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const preSongsRef = useRef<number>(0);

  const showToast = (kind: Toast["kind"], message: string) => {
    setToast({ kind, message });
    setTimeout(() => setToast(null), 5000);
  };

  const refreshSongs = useCallback(() => {
    fetchSongs()
      .then(setSongs)
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchSongs()
      .then(setSongs)
      .catch((e: Error & { status?: number }) => {
        if (e.status === 503) {
          router.replace("/candidates");
        } else {
          setError("Could not reach API — is the server running on localhost:8000?");
        }
      })
      .finally(() => setLoading(false));
  }, [router]);

  // Stop polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (handoffPollRef.current) clearInterval(handoffPollRef.current);
    };
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    preSongsRef.current = songs.length;
    try {
      await startGenerate();
    } catch (e) {
      setGenerating(false);
      showToast("error", e instanceof Error ? e.message : "Failed to start generate");
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const status = await getGenerateStatus();
        if (status.status === "done") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setGenerating(false);
          fetchSongs().then((updated: SongEntry[]) => {
            setSongs(updated);
            const newCount = updated.length - preSongsRef.current;
            showToast(
              "success",
              newCount > 0
                ? `${newCount} new song${newCount === 1 ? "" : "s"} generated`
                : "Generation complete"
            );
          });
        } else if (status.status === "error") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setGenerating(false);
          showToast("error", status.error ?? "Generation failed");
        }
      } catch {
        // transient poll failure — keep trying
      }
    }, 5000);
  };

  const handleSelect = async (song: SongEntry) => {
    setActivatingId(song.id);
    try {
      await activateSong(song.id);
      if (!song.initialized && song.proposal_path) {
        await initSong();
      }
      router.push("/candidates");
    } catch {
      setError(`Could not activate "${song.title}"`);
      setActivatingId(null);
    }
  };

  const handleHandoff = async (e: React.MouseEvent, song: SongEntry) => {
    e.stopPropagation();
    setHandoffingId(song.id);
    try {
      await activateSong(song.id);
      await startHandoff();
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Handoff failed");
      setHandoffingId(null);
      return;
    }
    handoffPollRef.current = setInterval(async () => {
      try {
        const job = await getHandoffStatus();
        if (job.status === "done") {
          clearInterval(handoffPollRef.current!);
          handoffPollRef.current = null;
          setHandoffingId(null);
          showToast("success", `Handoff complete for "${song.title}"`);
        } else if (job.status === "error") {
          clearInterval(handoffPollRef.current!);
          handoffPollRef.current = null;
          setHandoffingId(null);
          showToast("error", job.error ?? "Handoff failed");
        }
      } catch { /* transient */ }
    }, 2000);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center text-zinc-500 font-sans">
        Loading…
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 p-6 font-mono">
      <div className="flex items-start justify-between gap-4 mb-1">
        <h1 className="text-xl font-bold text-white tracking-tight">Songs</h1>
        {!error && (
          <button
            onClick={handleGenerate}
            disabled={generating || activatingId !== null}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {generating ? (
              <>
                <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Generating…
              </>
            ) : (
              "Generate New Song"
            )}
          </button>
        )}
      </div>
      <p className="text-zinc-500 text-xs font-sans mb-4">Select a song to review candidates</p>

      {toast && (
        <div
          className={`rounded p-3 mb-4 text-sm font-sans border ${
            toast.kind === "success"
              ? "bg-green-900/40 border-green-700 text-green-300"
              : "bg-red-900/40 border-red-700 text-red-300"
          }`}
        >
          {toast.message}
        </div>
      )}

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded p-3 mb-6 text-red-300 text-sm font-sans">
          {error}
        </div>
      )}

      {!error && songs.length === 0 && !generating && (
        <div className="text-zinc-500 font-sans text-sm">
          No songs found. Run shrinkwrap to scaffold production directories.
        </div>
      )}

      <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {songs.map(song => (
          <button
            key={song.id}
            onClick={() => handleSelect(song)}
            disabled={activatingId !== null || generating}
            className="text-left bg-zinc-900 border border-zinc-800 rounded-lg p-4 hover:border-zinc-600 hover:bg-zinc-800/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <span className="text-white font-semibold text-sm leading-snug">{song.title}</span>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                {song.has_decisions && (
                  <svg className="w-3.5 h-3.5 text-green-500" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Production decisions complete">
                    <title>Production decisions complete</title>
                    <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
                  </svg>
                )}
                {activatingId === song.id
                  ? <span className="text-zinc-500 text-xs font-sans">activating…</span>
                  : colorDot(song.rainbow_color)
                }
              </div>
            </div>
            <div className="text-zinc-500 text-xs font-sans mb-2 truncate">{song.thread_slug}</div>
            <div className="flex gap-3 text-xs font-sans text-zinc-400 flex-wrap items-center">
              {song.key && <span>{song.key}</span>}
              {song.bpm && <span>{song.bpm} BPM</span>}
              {song.singer && <span className="text-zinc-500">{song.singer}</span>}
              {!song.initialized && (
                <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 border border-zinc-700">
                  {activatingId === song.id ? "initializing…" : "not initialized"}
                </span>
              )}
            </div>
            {song.has_decisions && (
              <div className="mt-2 flex justify-end">
                <button
                  onClick={(e) => handleHandoff(e, song)}
                  disabled={handoffingId !== null || activatingId !== null || generating}
                  className="flex items-center gap-1.5 px-2 py-1 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:border-zinc-600 hover:text-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {handoffingId === song.id ? (
                    <>
                      <svg className="w-2.5 h-2.5 animate-spin" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                      </svg>
                      Handing off…
                    </>
                  ) : "Handoff to Logic"}
                </button>
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
