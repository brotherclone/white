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
import { assignSongToSide, fetchSides, fetchSongs, moveSongBetweenSides, removeSongFromSide } from "@/lib/api";
import { SideName, SidesResponse, SongEntry } from "@/lib/types";

const SIDE_NAMES: SideName[] = ["A", "B", "C", "D"];

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

function AvailableSongRow({ song }: { song: SongEntry }) {
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
      {disabled && <span className="text-[10px] text-zinc-600 shrink-0">no mix</span>}
    </div>
  );
}

function SideSongRow({
  side,
  songId,
  title,
  durationSeconds,
  onRemove,
}: {
  side: SideName;
  songId: string;
  title: string;
  durationSeconds: number;
  onRemove: () => void;
}) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `side:${side}:${songId}`,
    data: { songId, title, fromSide: side } satisfies DragPayload,
  });

  return (
    <div
      ref={setNodeRef}
      {...attributes}
      {...listeners}
      className={`px-3 py-2 rounded border border-zinc-700 bg-zinc-900 text-xs font-sans flex items-center justify-between gap-2 cursor-grab hover:border-zinc-500 ${
        isDragging ? "opacity-40" : ""
      }`}
    >
      <span className="truncate">{title}</span>
      <span className="flex items-center gap-2 shrink-0">
        <span className="text-zinc-500">{formatDuration(durationSeconds)}</span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
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
}: {
  side: SideName;
  limitSeconds: number;
  songs: { song_id: string; duration_seconds: number }[];
  totalSeconds: number;
  overLimit: boolean;
  titleFor: (songId: string) => string;
  onRemove: (songId: string) => void;
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

    try {
      if (payload.fromSide === null) {
        await assignSongToSide(toSide, payload.songId, toPosition);
      } else {
        await moveSongBetweenSides(payload.fromSide, payload.songId, toSide, toPosition);
      }
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Drop failed");
    }
  };

  const handleRemove = async (side: SideName, songId: string) => {
    try {
      await removeSongFromSide(side, songId);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Remove failed");
    }
  };

  const assignedIds = new Set(
    sides ? SIDE_NAMES.flatMap((s) => sides.sides[s].songs.map((song) => song.song_id)) : [],
  );
  const available = songs.filter((s) => !assignedIds.has(s.id));

  return (
    <div className="min-h-screen flex flex-col">
      <div className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4">
        <Link href="/" className="text-zinc-500 hover:text-zinc-300 text-xs font-sans transition-colors">
          ← home
        </Link>
        <h1 className="text-lg font-bold text-white tracking-tight">LP Side Sequencing</h1>
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
              <div className="flex flex-col gap-1.5">
                {available.map((song) => (
                  <AvailableSongRow key={song.id} song={song} />
                ))}
                {available.length === 0 && (
                  <div className="text-[11px] text-zinc-600 font-sans italic py-2">
                    All songs are assigned to a side
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
    </div>
  );
}
