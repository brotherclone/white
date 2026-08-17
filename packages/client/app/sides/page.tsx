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
  assignSongToSide, fetchMaterialProduced, fetchPlaylistConfig, fetchSides, fetchSongs,
  moveSongBetweenSides, removeSongFromSide, setPlaylistConfig, syncPlaylists,
} from "@/lib/api";
import { MaterialProduced, SideEntry, SideName, SideSong, SidesResponse, SongEntry } from "@/lib/types";
import { NotesButton, SongNotesModal } from "@/components/SongNotes";

const SIDE_NAMES: SideName[] = ["A", "B", "C", "D"];

const ALWAYS_EXCLUDED_LIFECYCLE_STATUSES = new Set<SongEntry["lifecycle_status"]>([
  "abandoned", "scrapped",
]);

// Merged tracks are only hidden from the pool when they have no audio to
// place on a side — a merged track that still has a mix should remain
// selectable regardless of the "show unmixed" toggle.
function isPoolEligible(s: SongEntry): boolean {
  if (ALWAYS_EXCLUDED_LIFECYCLE_STATUSES.has(s.lifecycle_status)) return false;
  if (s.lifecycle_status === "merged" && !s.has_mix) return false;
  return true;
}

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

function AvailableSongRow({
  song,
  onOpenNotes,
}: {
  song: SongEntry;
  onOpenNotes: (songId: string) => void;
}) {
  const disabled = !song.has_mix;
  const considered = song.lp_consideration !== "not_considered";
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
          : considered
            ? "border-blue-700 bg-blue-800 text-white cursor-grab hover:border-blue-500"
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
      className={`px-3 py-2 rounded border bg-blue-900/10 text-xs font-sans text-zinc-200 flex items-center justify-between gap-2 cursor-grab ${
        isOver ? "border-blue-400" : "border-blue-800 hover:border-blue-600"
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
  const [playlistDirInput, setPlaylistDirInput] = useState("");
  const [material, setMaterial] = useState<MaterialProduced | null>(null);
  const [savingPlaylistDir, setSavingPlaylistDir] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<{ rejects: number; review: number; wip: number } | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));

  useEffect(() => {
    fetchPlaylistConfig()
      .then((config) => setPlaylistDirInput(config.output_dir))
      .catch(() => {});
    fetchMaterialProduced().then(setMaterial).catch(() => {});
  }, []);

  const handleSavePlaylistDir = async () => {
    if (!playlistDirInput.trim()) return;
    setSavingPlaylistDir(true);
    setSyncError(null);
    try {
      await setPlaylistConfig(playlistDirInput.trim());
    } catch (e) {
      setSyncError(e instanceof Error ? e.message : "Failed to save output directory");
    } finally {
      setSavingPlaylistDir(false);
    }
  };

  const handleSync = async () => {
    setSyncing(true);
    setSyncError(null);
    setSyncResult(null);
    try {
      // Clicking this button blurs the directory input, which also fires
      // handleSavePlaylistDir (onBlur) — that save and this sync would race
      // if we didn't also save here first: awaiting it before syncing
      // guarantees the sync always uses the latest typed directory rather
      // than whatever was last persisted.
      if (playlistDirInput.trim()) {
        await setPlaylistConfig(playlistDirInput.trim());
      }
      const result = await syncPlaylists();
      setSyncResult(result);
    } catch (e) {
      setSyncError(e instanceof Error ? e.message : "Sync failed");
    } finally {
      setSyncing(false);
    }
  };

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
    .filter(isPoolEligible)
    .filter((s) => showUnmixed || s.has_mix)
    .filter((s) => !poolSearch || s.title.toLowerCase().includes(poolSearch.toLowerCase()));
  const unmixedHiddenCount = songs.filter(
    (s) => !assignedIds.has(s.id) && isPoolEligible(s) && !s.has_mix,
  ).length;
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
        <span className="ml-auto flex items-center gap-4">
          {sides && (
            <span className="text-xs font-sans text-zinc-500">
              Total: <span className="text-zinc-300">{formatDuration(totalUsedSeconds)}</span> / {formatDuration(totalTargetSeconds)} across {SIDE_NAMES.length} sides
            </span>
          )}
          {material && (
            <span className="text-xs font-sans text-zinc-500">
              Material produced: <span className="text-zinc-300">{formatDuration(material.total_seconds)}</span> / {formatDuration(material.target_seconds)}
            </span>
          )}
        </span>
      </div>

      <div className="border-b border-zinc-800 px-6 py-3 flex items-center gap-3">
        <span className="text-xs font-sans text-zinc-500 shrink-0">Playlist output:</span>
        <input
          type="text"
          value={playlistDirInput}
          onChange={(e) => setPlaylistDirInput(e.target.value)}
          onBlur={handleSavePlaylistDir}
          placeholder="/path/to/Listening"
          className="flex-1 max-w-xl bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
        />
        {savingPlaylistDir && <span className="text-[11px] text-zinc-600 font-sans">saving…</span>}
        <button
          onClick={handleSync}
          disabled={syncing}
          className="px-3 py-1 rounded border border-blue-700 bg-blue-950/40 text-xs font-sans text-blue-300 hover:bg-blue-950/70 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {syncing ? "Syncing…" : "Sync to Playlists"}
        </button>
        {syncResult && (
          <span className="text-xs font-sans text-zinc-400">
            Rejects: <span className="text-zinc-200">{syncResult.rejects}</span>, Review:{" "}
            <span className="text-zinc-200">{syncResult.review}</span>, White Album WiP:{" "}
            <span className="text-zinc-200">{syncResult.wip}</span>
          </span>
        )}
        {syncError && <span className="text-xs font-sans text-red-400">{syncError}</span>}
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
            <SongNotesModal song={notesSong} onClose={() => setNotesSongId(null)} onUpdated={refresh} />
          ) : null;
        })()}
    </div>
  );
}
