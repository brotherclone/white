"use client";

import { AvailabilityWindow, Collaborator } from "@/lib/types";

function isCurrentlyUnavailable(c: Collaborator): string | null {
  const today = new Date().toISOString().split("T")[0];
  for (const w of c.availability_windows) {
    if (w.unavailable_from <= today && today <= w.unavailable_until) {
      return w.reason ?? "Unavailable";
    }
  }
  return null;
}

function upcomingUnavailability(c: Collaborator): AvailabilityWindow | null {
  const today = new Date();
  const soon = new Date(today.getTime() + 14 * 24 * 60 * 60 * 1000);
  const todayStr = today.toISOString().split("T")[0];
  const soonStr = soon.toISOString().split("T")[0];
  for (const w of c.availability_windows) {
    if (w.unavailable_from > todayStr && w.unavailable_from <= soonStr) {
      return w;
    }
  }
  return null;
}

function formatDateRange(from: string, until: string): string {
  const fmt = (d: string) =>
    new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return `${fmt(from)} – ${fmt(until)}`;
}

const PLATFORM_LABELS: Record<string, string> = {
  airgigs: "AirGigs",
  soundbetter: "SoundBetter",
  direct: "Website",
  other: "Profile",
};

interface CollaboratorProfileCardProps {
  collaborator: Collaborator;
}

export default function CollaboratorProfileCard({ collaborator: c }: CollaboratorProfileCardProps) {
  const currentUnavail = isCurrentlyUnavailable(c);
  const upcoming = upcomingUnavailability(c);

  return (
    <div className="flex flex-col gap-3">
      {/* Avatar + name */}
      <div className="flex items-center gap-3">
        {c.photo_url ? (
          <img
            src={c.photo_url}
            alt={c.name}
            className="w-10 h-10 rounded-full object-cover border border-zinc-700"
          />
        ) : (
          <div className="w-10 h-10 rounded-full bg-zinc-700 border border-zinc-600 flex items-center justify-center text-sm font-semibold text-zinc-300">
            {c.name.charAt(0).toUpperCase()}
          </div>
        )}
        <div>
          <p className="text-sm font-semibold text-zinc-100">{c.name}</p>
          <p className="text-[10px] font-sans text-zinc-500">
            {c.roles.join(", ")}
          </p>
        </div>
      </div>

      {/* Availability warnings */}
      {currentUnavail && (
        <div className="px-3 py-2 rounded bg-yellow-900/30 border border-yellow-700 text-[10px] font-sans text-yellow-300">
          Currently unavailable: {currentUnavail}
        </div>
      )}
      {upcoming && !currentUnavail && (
        <div className="px-3 py-2 rounded bg-amber-900/30 border border-amber-700 text-[10px] font-sans text-amber-300">
          Upcoming unavailability: {formatDateRange(upcoming.unavailable_from, upcoming.unavailable_until)}
          {upcoming.reason && ` — ${upcoming.reason}`}
        </div>
      )}

      {/* Platform links */}
      {c.platforms.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {c.platforms.map((p, i) => (
            <a
              key={i}
              href={p.url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2 py-1 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
            >
              {PLATFORM_LABELS[p.platform] ?? p.platform}
            </a>
          ))}
          {c.website && (
            <a
              href={c.website}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2 py-1 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
            >
              Site
            </a>
          )}
        </div>
      )}

      {/* Social links */}
      {Object.keys(c.socials).length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(c.socials).map(([platform, url]) => (
            <a
              key={platform}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2 py-1 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200 transition-colors capitalize"
            >
              {platform}
            </a>
          ))}
        </div>
      )}

      {/* PRO */}
      {c.pro_affiliation !== "none" && (
        <p className="text-[10px] font-sans text-zinc-500">
          PRO: <span className="text-zinc-300 uppercase">{c.pro_affiliation}</span>
          {c.pro_number && <span className="text-zinc-500"> #{c.pro_number}</span>}
        </p>
      )}

      {/* Notes */}
      {c.notes && (
        <p className="text-[10px] font-sans text-zinc-500 italic">{c.notes}</p>
      )}
    </div>
  );
}
