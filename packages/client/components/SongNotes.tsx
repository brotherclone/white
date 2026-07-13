"use client";

import { useEffect, useState } from "react";
import {
  createDiaryEntry, fetchDiaryEntries, fetchSongMixInfo, setLpConsideration, songMixStreamUrl,
} from "@/lib/api";
import { DiaryEntry, SongEntry } from "@/lib/types";

export function SongNotesModal({
  song,
  onClose,
  onUpdated,
}: {
  song: SongEntry;
  onClose: () => void;
  onUpdated: () => void;
}) {
  const [mixInfo, setMixInfo] = useState<{
    has_mix: boolean;
    mix_file: string | null;
    duration_seconds: number | null;
  } | null>(null);
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [loadingEntries, setLoadingEntries] = useState(true);
  const [phase, setPhase] = useState("");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [candidatePending, setCandidatePending] = useState(false);

  const handleToggleCandidate = async () => {
    setCandidatePending(true);
    try {
      await setLpConsideration(song.id, song.lp_consideration === "candidate" ? "not_considered" : "candidate");
      onUpdated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update candidate status");
    } finally {
      setCandidatePending(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    setMixInfo(null);
    setLoadingEntries(true);

    fetchSongMixInfo(song.id).then((info) => {
      if (!cancelled) setMixInfo(info);
    });
    fetchDiaryEntries(song.production_slug)
      .then((fetched) => {
        if (!cancelled) setEntries([...fetched].reverse());
      })
      .catch(() => {
        if (!cancelled) setEntries([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingEntries(false);
      });

    return () => {
      cancelled = true;
    };
  }, [song.id, song.production_slug]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const created = await createDiaryEntry(song.production_slug, {
        song_slug: song.production_slug,
        author: "gabriel",
        phase: phase || null,
        title: title || null,
        body,
        tags: [],
        metadata: {},
      });
      setEntries((prev) => [created, ...prev]);
      setPhase("");
      setTitle("");
      setBody("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-lg mx-4 flex flex-col max-h-[85vh] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 flex-shrink-0">
          <span className="text-sm font-semibold text-white font-sans truncate">{song.title}</span>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 text-lg leading-none transition-colors"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="px-5 py-3 border-b border-zinc-800/60 flex-shrink-0 flex items-center gap-3">
          <div className="flex-1 min-w-0">
            {mixInfo === null ? (
              <p className="text-[11px] font-sans text-zinc-600 italic">Loading mix…</p>
            ) : mixInfo.has_mix ? (
              <audio controls src={songMixStreamUrl(song.id)} className="w-full h-9" />
            ) : (
              <p className="text-[11px] font-sans text-zinc-600 italic">No mix file yet</p>
            )}
          </div>
          {song.lp_consideration === "placed" ? (
            <span className="text-[10px] font-sans px-2 py-1 rounded border border-blue-800 bg-blue-900/30 text-blue-400 shrink-0">
              Placed
            </span>
          ) : (
            <button
              onClick={handleToggleCandidate}
              disabled={candidatePending}
              className={`text-[10px] font-sans px-2 py-1 rounded border shrink-0 transition-colors disabled:opacity-50 ${
                song.lp_consideration === "candidate"
                  ? "border-blue-800 bg-blue-900/30 text-blue-400 hover:bg-blue-900/50"
                  : "border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
              }`}
            >
              {song.lp_consideration === "candidate" ? "★ Candidate" : "Mark as Candidate"}
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-3 flex flex-col gap-2">
          {loadingEntries ? (
            <p className="text-[11px] font-sans text-zinc-600 italic">Loading entries…</p>
          ) : entries.length === 0 ? (
            <p className="text-[11px] font-sans text-zinc-600 italic">No diary entries yet</p>
          ) : (
            entries.map((entry) => (
              <div key={entry.id} className="border border-zinc-800 rounded px-3 py-2">
                <div className="flex items-center justify-between gap-2 text-[10px] font-sans text-zinc-500">
                  <span>
                    {entry.author}
                    {entry.phase ? ` · ${entry.phase}` : ""}
                  </span>
                  <span>{new Date(entry.created_at).toLocaleString()}</span>
                </div>
                {entry.title && (
                  <p className="text-xs font-sans font-semibold text-zinc-200 mt-1">{entry.title}</p>
                )}
                <p className="text-xs font-sans text-zinc-300 mt-1 whitespace-pre-wrap">{entry.body}</p>
              </div>
            ))
          )}
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3 px-5 py-4 border-t border-zinc-800 flex-shrink-0">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-sans text-zinc-500 uppercase tracking-wider">Phase</span>
            <input
              type="text"
              value={phase}
              onChange={(e) => setPhase(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-sans text-zinc-500 uppercase tracking-wider">Title</span>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="optional headline"
              className="bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-sans text-zinc-500 uppercase tracking-wider">Body</span>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              required
              rows={4}
              placeholder="what happened?"
              className="bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 resize-none transition-colors"
            />
          </label>
          {error && <p className="text-[10px] font-sans text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={saving}
            className="w-full py-2 text-sm font-sans font-semibold rounded-lg bg-violet-700 hover:bg-violet-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? "Saving…" : "Save entry"}
          </button>
        </form>
      </div>
    </div>
  );
}

export function NotesButton({
  title,
  onOpen,
  label,
}: {
  title: string;
  onOpen: () => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onOpen();
      }}
      aria-label={label ? `${label} for ${title}` : `Listen and add notes for ${title}`}
      title={label ? `${label} / diary notes` : "Listen / diary notes"}
      className={`shrink-0 flex items-center justify-center gap-2 border border-[var(--ef-gray)] bg-transparent text-[#c9c9c9] hover:text-[var(--ef-orange)] hover:border-[var(--ef-orange)] transition-[color,border-color] duration-500 ease-in-out ${
        label ? "h-[34px] px-3 text-xs font-sans" : "w-[34px] h-[34px]"
      }`}
    >
      <i className="fa-solid fa-feather-pointed text-[15px]" aria-hidden="true" />
      {label && <span>{label}</span>}
    </button>
  );
}
