"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchSongs, activateSong, initSong } from "@/lib/api";
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
  ideation:    "Ideation",
  generation:  "Generation",
  composition: "Composition",
  production:  "Production",
  mixing:      "Mixing",
  complete:    "Complete",
  invalid:     "Invalid",
};

const STAGE_BADGE_CLS: Record<SongEntry["stage"], string> = {
  ideation:    "bg-zinc-800 text-zinc-500 border-zinc-700",
  generation:  "bg-blue-900/40 text-blue-300 border-blue-800",
  composition: "bg-violet-900/40 text-violet-300 border-violet-700",
  production:  "bg-orange-900/40 text-orange-300 border-orange-800",
  mixing:      "bg-cyan-900/40 text-cyan-300 border-cyan-800",
  complete:    "bg-green-900/40 text-green-300 border-green-700",
  invalid:     "bg-red-900/40 text-red-300 border-red-800",
};

const ALL_SONG_STAGES = Object.keys(STAGE_LABELS) as SongEntry["stage"][];

type Toast = { kind: "success" | "error"; message: string };
type StageFilter = SongEntry["stage"] | "all" | "stub";

export default function SongBrowserPage() {
  const router = useRouter();
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const [stageFilter, setStageFilter] = useState<StageFilter>("all");

  const countsByStage = useMemo(
    () => songs.reduce<Partial<Record<SongEntry["stage"], number>>>((acc, s) => {
      acc[s.stage] = (acc[s.stage] ?? 0) + 1;
      return acc;
    }, {}),
    [songs],
  );

  const stubCount = useMemo(() => songs.filter(s => s.stub).length, [songs]);

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

  const handleSelect = async (song: SongEntry) => {
    if (song.stage === "invalid") {
      showToast("error", "Song metadata is invalid — run migration to repair");
      return;
    }
    setActivatingId(song.id);
    try {
      await activateSong(song.id);
      if (song.stage === "ideation" && song.proposal_path) {
        await initSong();
      }
      if (["composition", "production", "mixing", "complete"].includes(song.stage)) {
        router.push("/board");
      } else {
        router.push("/candidates");
      }
    } catch {
      setError(`Could not activate "${song.title}"`);
      setActivatingId(null);
    }
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
      <p className="text-zinc-500 text-xs font-sans mb-3">Select a song to continue</p>

      {/* Stage filter */}
      {!error && songs.length > 0 && (
        <div className="flex gap-1.5 flex-wrap mb-4">
          {(["all", ...ALL_SONG_STAGES.filter(s => s !== "invalid"), "stub", "invalid"] as StageFilter[]).map(s => {
            const count = s === "all" ? songs.length : s === "stub" ? stubCount : (countsByStage[s as SongEntry["stage"]] ?? 0);
            const active = stageFilter === s;
            return (
              <button
                key={s}
                type="button"
                aria-pressed={active}
                onClick={() => setStageFilter(s)}
                className={`px-2.5 py-1 text-[10px] font-sans border transition-colors ${
                  active
                    ? "bg-zinc-200 text-zinc-900 border-zinc-200"
                    : "bg-zinc-900 text-zinc-400 border-zinc-700 hover:border-zinc-500 hover:text-zinc-200"
                }`}
              >
                {s === "all" ? "all" : s === "stub" ? "stub" : STAGE_LABELS[s as SongEntry["stage"]].toLowerCase()} <span className="text-zinc-600">{count}</span>
              </button>
            );
          })}
        </div>
      )}

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
        {songs.filter(s => stageFilter === "all" || (stageFilter === "stub" ? s.stub : s.stage === stageFilter)).map(song => (
          <div
            key={song.id}
            onClick={() => activatingId === null && handleSelect(song)}
            role="button"
            tabIndex={activatingId !== null ? -1 : 0}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (activatingId === null) handleSelect(song); } }}
            className={`text-left bg-zinc-900 border rounded-lg p-4 transition-colors focus:outline-none focus:ring-2 focus:ring-[#EF7143] ${
              song.stage === "invalid"
                ? "border-red-900 hover:border-red-700 cursor-default"
                : activatingId !== null
                ? "border-zinc-800 opacity-50 cursor-not-allowed"
                : "border-zinc-800 hover:border-zinc-600 hover:bg-zinc-800/80 cursor-pointer"
            }`}
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <span className="text-white font-semibold text-sm leading-snug">{song.title}</span>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                {activatingId === song.id
                  ? <span className="text-zinc-500 text-xs font-sans">activating…</span>
                  : colorDot(song.rainbow_color)
                }
              </div>
            </div>
            <div className="text-zinc-500 text-xs font-sans mb-2 truncate">{song.thread_slug}</div>
            <div className="flex gap-3 text-xs font-sans text-zinc-400 flex-wrap items-center">
              {song.stub
                ? <span className="text-amber-500/80 italic">Incomplete metadata</span>
                : <>
                    {song.key && <span>{song.key}</span>}
                    {song.bpm && <span>{song.bpm}{song.time_sig ? ` BPM · ${song.time_sig}` : " BPM"}</span>}
                    {song.singer && <span className="text-zinc-500">{song.singer}</span>}
                  </>
              }
              {song.has_mix && <span title="Mix attached" className="text-zinc-400">♫</span>}
              <span className={`ml-auto text-[10px] px-1.5 py-0.5 rounded border ${STAGE_BADGE_CLS[song.stage]}`}>
                {activatingId === song.id && song.stage === "ideation"
                  ? "initializing…"
                  : STAGE_LABELS[song.stage]}
              </span>
            </div>
            {(song.schema_version !== "2.0.0" || song.stub) && (
              <div className="mt-1.5 flex items-center gap-1.5">
                <span className="text-[9px] font-sans px-1 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-zinc-500">
                  v{song.schema_version}
                </span>
                {song.stub && (
                  <span className="text-[9px] font-sans px-1 py-0.5 rounded bg-amber-900/30 border border-amber-800 text-amber-400">
                    stub
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
