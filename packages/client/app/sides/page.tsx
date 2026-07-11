"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  PointerSensor,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  assignSongToSide, createDiaryEntry, fetchDiaryEntries, fetchSides, fetchSongMixInfo,
  fetchSongs, moveSongBetweenSides, removeSongFromSide, songMixStreamUrl,
} from "@/lib/api";
import { DiaryEntry, SideEntry, SideName, SideSong, SidesResponse, SongEntry } from "@/lib/types";

const SIDE_NAMES: SideName[] = ["A", "B", "C", "D"];

function sideEntryFromSongs(songs: SideSong[], limitSeconds: number): SideEntry {
  const total = songs.reduce((sum, s) => sum + s.duration_seconds, 0);
  return { songs, total_seconds: total, over_limit: total > limitSeconds };
}

/** Move songId to toSide/toPosition in a local copy of `sides`, recomputing totals. */
function applyLocalPlacement(
  sides: SidesResponse,
  songId: string,
  toSide: SideName,
  toPosition: number,
  knownDuration: number | null,
): SidesResponse {
  let duration = knownDuration;
  const withoutSong: Record<SideName, SideSong[]> = {} as Record<SideName, SideSong[]>;
  for (const name of SIDE_NAMES) {
    const existing = sides.sides[name].songs.find((s) => s.song_id === songId);
    if (existing && duration === null) duration = existing.duration_seconds;
    withoutSong[name] = sides.sides[name].songs.filter((s) => s.song_id !== songId);
  }

  const targetSongs = [...withoutSong[toSide]];
  const position = Math.max(0, Math.min(toPosition, targetSongs.length));
  targetSongs.splice(position, 0, { song_id: songId, duration_seconds: duration ?? 0 });
  withoutSong[toSide] = targetSongs;

  const newSides = {} as Record<SideName, SideEntry>;
  for (const name of SIDE_NAMES) {
    newSides[name] = sideEntryFromSongs(withoutSong[name], sides.side_limit_seconds);
  }
  return { side_limit_seconds: sides.side_limit_seconds, sides: newSides };
}

/** Remove songId from `side` in a local copy of `sides`, recomputing totals. */
function applyLocalRemoval(sides: SidesResponse, side: SideName, songId: string): SidesResponse {
  const remaining = sides.sides[side].songs.filter((s) => s.song_id !== songId);
  return {
    ...sides,
    sides: {
      ...sides.sides,
      [side]: sideEntryFromSongs(remaining, sides.side_limit_seconds),
    },
  };
}

function formatDuration(seconds: number): string {
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

interface DragPayload {
  songId: string;
  title: string;
  fromSide: SideName | null;
}

function SongNotesModal({ song, onClose }: { song: SongEntry; onClose: () => void }) {
  const [mixInfo, setMixInfo] = useState<{
    has_mix: boolean;
    mix_file: string | null;
    duration_seconds: number | null;
  } | null>(null);
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [loadingEntries, setLoadingEntries] = useState(true);
  const [author, setAuthor] = useState("");
  const [phase, setPhase] = useState("");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSongMixInfo(song.id).then(setMixInfo);
    fetchDiaryEntries(song.production_slug)
      .then((fetched) => setEntries([...fetched].reverse()))
      .catch(() => setEntries([]))
      .finally(() => setLoadingEntries(false));
  }, [song.id, song.production_slug]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const created = await createDiaryEntry(song.production_slug, {
        song_slug: song.production_slug,
        author,
        phase: phase || null,
        title: title || null,
        body,
        tags: [],
        metadata: {},
      });
      setEntries((prev) => [created, ...prev]);
      setAuthor("");
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

        <div className="px-5 py-3 border-b border-zinc-800/60 flex-shrink-0">
          {mixInfo === null ? (
            <p className="text-[11px] font-sans text-zinc-600 italic">Loading mix…</p>
          ) : mixInfo.has_mix ? (
            <audio controls src={songMixStreamUrl(song.id)} className="w-full h-9" />
          ) : (
            <p className="text-[11px] font-sans text-zinc-600 italic">No mix file yet</p>
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
          <div className="flex gap-3">
            <label className="flex flex-col gap-1 flex-1">
              <span className="text-[10px] font-sans text-zinc-500 uppercase tracking-wider">Author</span>
              <input
                type="text"
                value={author}
                onChange={(e) => setAuthor(e.target.value)}
                required
                placeholder="gabriel"
                className="bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
              />
            </label>
            <label className="flex flex-col gap-1 flex-1">
              <span className="text-[10px] font-sans text-zinc-500 uppercase tracking-wider">Phase</span>
              <input
                type="text"
                value={phase}
                onChange={(e) => setPhase(e.target.value)}
                className="bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
              />
            </label>
          </div>
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

function NotesButton({ title, onOpen }: { title: string; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onOpen();
      }}
      aria-label={`Listen and add notes for ${title}`}
      title="Listen / diary notes"
      className="w-[34px] h-[34px] shrink-0 flex items-center justify-center border border-[var(--ef-gray)] bg-transparent text-[#c9c9c9] hover:text-[var(--ef-orange)] hover:border-[var(--ef-orange)] transition-[color,border-color] duration-500 ease-in-out"
    >
      <i className="fa-solid fa-feather-pointed text-[15px]" aria-hidden="true" />
    </button>
  );
}

function AvailableSongRow({
  song,
  onOpenNotes,
}: {
  song: SongEntry;
  onOpenNotes: (songId: string) => void;
}) {
  const disabled = !song.has_mix;
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `avail:${song.id}`,
    data: { songId: song.id, title: song.title, fromSide: null } satisfies DragPayload,
    disabled,
  });

  return (
    <div
      ref={setNodeRef}
      {...(disabled ? {} : { ...attributes, ...listeners })}
      className={`px-3 py-2 rounded border text-xs font-sans flex items-center justify-between gap-2 ${
        disabled
          ? "border-zinc-800 text-zinc-600 cursor-not-allowed"
          : "border-zinc-700 text-zinc-200 cursor-grab hover:border-zinc-500"
      } ${isDragging ? "opacity-40" : ""}`}
      title={disabled ? "No mix file — cannot be sequenced" : "Drag onto a side"}
    >
      <span className="truncate">{song.title}</span>
      <span className="flex items-center gap-2 shrink-0">
        {disabled && <span className="text-[10px] text-zinc-600">no mix</span>}
        <NotesButton title={song.title} onOpen={() => onOpenNotes(song.id)} />
      </span>
    </div>
  );
}

function SideSongRow({
  side,
  songId,
  title,
  durationSeconds,
  onRemove,
  onOpenNotes,
}: {
  side: SideName;
  songId: string;
  title: string;
  durationSeconds: number;
  onRemove: () => void;
  onOpenNotes: (songId: string) => void;
}) {
  const { attributes, listeners, setNodeRef: setDragRef, isDragging } = useDraggable({
    id: `side:${side}:${songId}`,
    data: { songId, title, fromSide: side } satisfies DragPayload,
  });
  // Also a drop target so releasing directly on this row inserts before it,
  // instead of always falling back to appending at the end of the side.
  const { setNodeRef: setDropRef, isOver } = useDroppable({ id: `side:${side}:${songId}` });
  const setRefs = (node: HTMLDivElement | null) => {
    setDragRef(node);
    setDropRef(node);
  };

  return (
    <div
      ref={setRefs}
      {...attributes}
      {...listeners}
      className={`px-3 py-2 rounded border bg-zinc-900 text-xs font-sans text-zinc-200 flex items-center justify-between gap-2 cursor-grab ${
        isOver ? "border-blue-500" : "border-zinc-700 hover:border-zinc-500"
      } ${isDragging ? "opacity-40" : ""}`}
    >
      <span className="truncate">{title}</span>
      <span className="flex items-center gap-2 shrink-0">
        <span className="text-zinc-500">{formatDuration(durationSeconds)}</span>
        <NotesButton title={title} onOpen={() => onOpenNotes(songId)} />
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label={`Remove ${title} from side ${side}`}
          className="text-zinc-600 hover:text-red-400 transition-colors"
          title="Remove from side"
        >
          ×
        </button>
      </span>
    </div>
  );
}

function SideColumn({
  side,
  limitSeconds,
  songs,
  totalSeconds,
  overLimit,
  titleFor,
  onRemove,
  onOpenNotes,
}: {
  side: SideName;
  limitSeconds: number;
  songs: { song_id: string; duration_seconds: number }[];
  totalSeconds: number;
  overLimit: boolean;
  titleFor: (songId: string) => string;
  onRemove: (songId: string) => void;
  onOpenNotes: (songId: string) => void;
}) {
  const { setNodeRef, isOver } = useDroppable({ id: `sidearea:${side}` });

  return (
    <div
      ref={setNodeRef}
      className={`flex-1 min-w-[220px] rounded border p-3 flex flex-col gap-2 ${
        isOver ? "border-blue-500 bg-blue-950/20" : "border-zinc-800"
      }`}
    >
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-white">Side {side}</h2>
        <span className={`text-xs font-sans ${overLimit ? "text-red-400" : "text-zinc-500"}`}>
          {formatDuration(totalSeconds)} / {formatDuration(limitSeconds)}
        </span>
      </div>
      <div className="flex flex-col gap-1.5 min-h-[40px]">
        {songs.map((s) => (
          <SideSongRow
            key={s.song_id}
            side={side}
            songId={s.song_id}
            title={titleFor(s.song_id)}
            durationSeconds={s.duration_seconds}
            onRemove={() => onRemove(s.song_id)}
            onOpenNotes={onOpenNotes}
          />
        ))}
        {songs.length === 0 && (
          <div className="text-[11px] text-zinc-600 font-sans italic py-2">Drop a mixed song here</div>
        )}
      </div>
    </div>
  );
}

export default function SidesPage() {
  const [songs, setSongs] = useState<SongEntry[]>([]);
  const [sides, setSides] = useState<SidesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeDrag, setActiveDrag] = useState<DragPayload | null>(null);
  const [poolSearch, setPoolSearch] = useState("");
  const [showUnmixed, setShowUnmixed] = useState(false);
  const [notesSongId, setNotesSongId] = useState<string | null>(null);

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));

  const refresh = useCallback(() => {
    Promise.all([fetchSongs(), fetchSides()])
      .then(([songsRes, sidesRes]) => {
        setSongs(songsRes);
        setSides(sidesRes);
        setError(null);
      })
      .catch((e) => setError(e.message ?? "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const titleFor = useCallback(
    (songId: string) => songs.find((s) => s.id === songId)?.title ?? songId,
    [songs],
  );

  const handleDragStart = (event: DragStartEvent) => {
    setActiveDrag(event.active.data.current as DragPayload);
  };

  const handleDragEnd = async (event: DragEndEvent) => {
    setActiveDrag(null);
    const { active, over } = event;
    if (!over || !sides) return;

    const payload = active.data.current as DragPayload;
    const overId = String(over.id);
    let toSide: SideName | null = null;
    let toPosition = 0;

    if (overId.startsWith("sidearea:")) {
      toSide = overId.split(":")[1] as SideName;
      toPosition = sides.sides[toSide].songs.length;
    } else if (overId.startsWith("side:")) {
      const [, sideName, overSongId] = overId.split(":");
      toSide = sideName as SideName;
      const idx = sides.sides[toSide].songs.findIndex((s) => s.song_id === overSongId);
      toPosition = idx < 0 ? sides.sides[toSide].songs.length : idx;
    }
    if (!toSide) return;

    const previousSides = sides;
    const knownDuration =
      payload.fromSide !== null
        ? (sides.sides[payload.fromSide].songs.find((s) => s.song_id === payload.songId)
            ?.duration_seconds ?? null)
        : null;

    // Update the UI immediately so the item disappears from its source on release;
    // reconcile with the server's authoritative (duration-accurate) response after.
    setSides(applyLocalPlacement(sides, payload.songId, toSide, toPosition, knownDuration));

    try {
      const response =
        payload.fromSide === null
          ? await assignSongToSide(toSide, payload.songId, toPosition)
          : await moveSongBetweenSides(payload.fromSide, payload.songId, toSide, toPosition);
      const resolvedSide = toSide;
      setSides((current) =>
        current
          ? {
              ...current,
              sides: {
                ...current.sides,
                [resolvedSide]: sideEntryFromSongs(response.songs, current.side_limit_seconds),
              },
            }
          : current,
      );
      setError(null);
      fetchSongs().then(setSongs).catch(() => {});
    } catch (e) {
      setSides(previousSides);
      setError(e instanceof Error ? e.message : "Drop failed");
    }
  };

  const handleRemove = async (side: SideName, songId: string) => {
    if (!sides) return;
    const previousSides = sides;
    setSides(applyLocalRemoval(sides, side, songId));

    try {
      await removeSongFromSide(side, songId);
      setError(null);
      fetchSongs().then(setSongs).catch(() => {});
    } catch (e) {
      setSides(previousSides);
      setError(e instanceof Error ? e.message : "Remove failed");
    }
  };

  const assignedIds = new Set(
    sides ? SIDE_NAMES.flatMap((s) => sides.sides[s].songs.map((song) => song.song_id)) : [],
  );
  const available = songs
    .filter((s) => !assignedIds.has(s.id))
    .filter((s) => showUnmixed || s.has_mix)
    .filter((s) => !poolSearch || s.title.toLowerCase().includes(poolSearch.toLowerCase()));
  const unmixedHiddenCount = songs.filter((s) => !assignedIds.has(s.id) && !s.has_mix).length;
  const allAssigned = songs.length > 0 && songs.every((s) => assignedIds.has(s.id));

  const totalUsedSeconds = sides
    ? SIDE_NAMES.reduce((sum, s) => sum + sides.sides[s].total_seconds, 0)
    : 0;
  const totalTargetSeconds = sides ? sides.side_limit_seconds * SIDE_NAMES.length : 0;

  return (
    <div className="min-h-screen flex flex-col">
      <div className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4">
        <Link href="/" className="text-zinc-500 hover:text-zinc-300 text-xs font-sans transition-colors">
          ← home
        </Link>
        <h1 className="text-lg font-bold text-white tracking-tight">LP Side Sequencing</h1>
        {sides && (
          <span className="ml-auto text-xs font-sans text-zinc-500">
            Total: <span className="text-zinc-300">{formatDuration(totalUsedSeconds)}</span> / {formatDuration(totalTargetSeconds)} across {SIDE_NAMES.length} sides
          </span>
        )}
      </div>

      {error && (
        <div className="mx-6 mt-4 px-3 py-2 rounded border border-red-800 bg-red-950/30 text-red-300 text-xs font-sans">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-6 text-zinc-500 text-sm font-sans">Loading…</div>
      ) : (
        <DndContext sensors={sensors} onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
          <div className="flex-1 flex gap-4 p-6">
            <div className="w-64 flex flex-col gap-2">
              <h2 className="text-sm font-bold text-white">Available</h2>
              <input
                type="text"
                value={poolSearch}
                onChange={(e) => setPoolSearch(e.target.value)}
                placeholder="Search song title…"
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
              />
              {unmixedHiddenCount > 0 && (
                <label className="flex items-center gap-1.5 text-[11px] font-sans text-zinc-500 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showUnmixed}
                    onChange={(e) => setShowUnmixed(e.target.checked)}
                    className="accent-blue-600"
                  />
                  Show {unmixedHiddenCount} without a mix
                </label>
              )}
              <div className="flex flex-col gap-1.5">
                {available.map((song) => (
                  <AvailableSongRow key={song.id} song={song} onOpenNotes={setNotesSongId} />
                ))}
                {available.length === 0 && (
                  <div className="text-[11px] text-zinc-600 font-sans italic py-2">
                    {allAssigned ? "All songs are assigned to a side" : "No matching songs"}
                  </div>
                )}
              </div>
            </div>

            {sides &&
              SIDE_NAMES.map((side) => (
                <SideColumn
                  key={side}
                  side={side}
                  limitSeconds={sides.side_limit_seconds}
                  songs={sides.sides[side].songs}
                  totalSeconds={sides.sides[side].total_seconds}
                  overLimit={sides.sides[side].over_limit}
                  titleFor={titleFor}
                  onRemove={(songId) => handleRemove(side, songId)}
                  onOpenNotes={setNotesSongId}
                />
              ))}
          </div>

          <DragOverlay>
            {activeDrag && (
              <div className="px-3 py-2 rounded border border-blue-500 bg-zinc-900 text-xs font-sans text-zinc-200 shadow-lg">
                {activeDrag.title}
              </div>
            )}
          </DragOverlay>
        </DndContext>
      )}

      {notesSongId &&
        (() => {
          const notesSong = songs.find((s) => s.id === notesSongId);
          return notesSong ? (
            <SongNotesModal song={notesSong} onClose={() => setNotesSongId(null)} />
          ) : null;
        })()}
    </div>
  );
}
