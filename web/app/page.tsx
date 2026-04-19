"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchSongs, activateSong } from "@/lib/api";
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

export default function SongIndexPage() {
  const router = useRouter();
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    setActivatingId(song.id);
    try {
      await activateSong(song.id);
      router.push("/candidates");
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
      <h1 className="text-xl font-bold text-white tracking-tight mb-1">Songs</h1>
      <p className="text-zinc-500 text-xs font-sans mb-6">Select a song to review candidates</p>

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
          <button
            key={song.id}
            onClick={() => handleSelect(song)}
            disabled={activatingId !== null}
            className="text-left bg-zinc-900 border border-zinc-800 rounded-lg p-4 hover:border-zinc-600 hover:bg-zinc-800/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <span className="text-white font-semibold text-sm leading-snug">{song.title}</span>
              {activatingId === song.id
                ? <span className="text-zinc-500 text-xs font-sans flex-shrink-0">activating…</span>
                : colorDot(song.rainbow_color)
              }
            </div>
            <div className="text-zinc-500 text-xs font-sans mb-2 truncate">{song.thread_slug}</div>
            <div className="flex gap-3 text-xs font-sans text-zinc-400 flex-wrap">
              {song.key && <span>{song.key}</span>}
              {song.bpm && <span>{song.bpm} BPM</span>}
              {song.singer && <span className="text-zinc-500">{song.singer}</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
