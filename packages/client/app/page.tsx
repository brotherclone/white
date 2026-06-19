"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchSongs, activateSong, initSong, startHandoff, getHandoffStatus } from "@/lib/api";
import { SongEntry } from "@/lib/types";

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

const STAGE_LABELS: Record<SongEntry["stage"], string> = {
  ideation: "Ideation",
  generation: "Generation",
  composition: "Composition",
};

type Toast = { kind: "success" | "error"; message: string };

export default function SongBrowserPage() {
  const router = useRouter();
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [handoffingId, setHandoffingId] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const handoffPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const showToast = (kind: Toast["kind"], message: string) => {
    setToast({ kind, message });
    setTimeout(() => setToast(null), 5000);
  };

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

  useEffect(() => {
    return () => {
      if (handoffPollRef.current) clearInterval(handoffPollRef.current);
    };
  }, []);

  const handleSelect = async (song: SongEntry) => {
    setActivatingId(song.id);
    try {
      await activateSong(song.id);
      if (song.stage === "ideation" && song.proposal_path) {
        await initSong();
      }
      if (song.stage === "composition") {
        router.push("/board");
      } else {
        router.push("/candidates");
      }
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
        <div className="flex items-center gap-2">
          <Link
            href="/collaborators"
            className="px-3 py-1.5 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 transition-colors"
          >
            Collaborators
          </Link>
          {!error && (
            <Link
              href="/agent"
              className="px-3 py-1.5 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 transition-colors"
            >
              Run Agent
            </Link>
          )}
        </div>
      </div>
      <p className="text-zinc-500 text-xs font-sans mb-4">Select a song to continue</p>

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

      {!error && songs.length === 0 && (
        <div className="text-zinc-500 font-sans text-sm">
          No songs found. Run shrinkwrap to scaffold production directories.
        </div>
      )}

      <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {songs.map(song => (
          <div
            key={song.id}
            onClick={() => activatingId === null && handleSelect(song)}
            role="button"
            tabIndex={activatingId !== null ? -1 : 0}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (activatingId === null) handleSelect(song); } }}
            className={`text-left bg-zinc-900 border border-zinc-800 rounded-lg p-4 hover:border-zinc-600 hover:bg-zinc-800/80 transition-colors focus:outline-none focus:ring-2 focus:ring-[#EF7143] ${activatingId !== null ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <span className="text-white font-semibold text-sm leading-snug">{song.title}</span>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                {song.has_decisions && song.stage !== "composition" && (
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
              {song.bpm && <span>{song.bpm}{song.time_sig ? ` BPM · ${song.time_sig}` : " BPM"}</span>}
              {song.singer && <span className="text-zinc-500">{song.singer}</span>}
              {song.has_mix && <span title="Mix attached" className="text-zinc-400">♫</span>}
              <span className={`ml-auto text-[10px] px-1.5 py-0.5 rounded border ${
                song.stage === "composition"
                  ? "bg-violet-900/40 text-violet-300 border-violet-700"
                  : song.stage === "generation"
                  ? "bg-blue-900/40 text-blue-300 border-blue-800"
                  : "bg-zinc-800 text-zinc-500 border-zinc-700"
              }`}>
                {activatingId === song.id && song.stage === "ideation"
                  ? "initializing…"
                  : STAGE_LABELS[song.stage]}
              </span>
            </div>
            {song.has_decisions && song.stage !== "composition" && (
              <div className="mt-2 flex justify-end">
                <button
                  onClick={(e) => handleHandoff(e, song)}
                  disabled={handoffingId !== null || activatingId !== null}
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
          </div>
        ))}
      </div>
    </div>
  );
}
