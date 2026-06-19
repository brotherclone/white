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
      {/* Avatar + name — square per brand (no rounded-full) */}
      <div className="flex items-center gap-3">
        {c.photo_url ? (
          <img
            src={c.photo_url}
            alt={c.name}
            className="w-10 h-10 object-cover border border-[#333] flex-shrink-0"
          />
        ) : (
          <div className="w-10 h-10 bg-[#1a1a1a] border border-[#333] flex items-center justify-center text-sm font-display font-bold text-[#9b9b9b] flex-shrink-0">
            {c.name.charAt(0).toUpperCase()}
          </div>
        )}
        <div>
          <p className="text-sm font-display font-bold text-[#f6f6f6]">{c.name}</p>
          <p className="text-[10px] font-sans text-[#9b9b9b]">{c.roles.join(", ")}</p>
        </div>
      </div>

      {/* Availability warnings — orange = current, yellow = upcoming */}
      {currentUnavail && (
        <div className="px-3 py-2 text-[10px] font-sans" style={{ background: "#EF7143", color: "#000" }}>
          currently unavailable: {currentUnavail}
        </div>
      )}
      {upcoming && !currentUnavail && (
        <div className="px-3 py-2 text-[10px] font-sans" style={{ background: "#ffff00", color: "#383838" }}>
          upcoming: {formatDateRange(upcoming.unavailable_from, upcoming.unavailable_until)}
          {upcoming.reason && ` — ${upcoming.reason}`}
        </div>
      )}

      {/* Platform links — square, hairline border */}
      {c.platforms.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {c.platforms.map((p, i) => (
            <a
              key={i}
              href={p.url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2 py-1 text-[10px] font-sans bg-[#000] border border-[#333] text-[#cbcbcb] hover:text-[#f6f6f6] transition-colors"
              style={{ transitionDuration: "300ms" }}
            >
              {PLATFORM_LABELS[p.platform] ?? p.platform}
            </a>
          ))}
          {c.website && (
            <a
              href={c.website}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2 py-1 text-[10px] font-sans bg-[#000] border border-[#333] text-[#cbcbcb] hover:text-[#f6f6f6] transition-colors"
              style={{ transitionDuration: "300ms" }}
            >
              site
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
              className="px-2 py-1 text-[10px] font-sans bg-[#000] border border-[#333] text-[#9b9b9b] hover:text-[#f6f6f6] transition-colors capitalize"
              style={{ transitionDuration: "300ms" }}
            >
              {platform}
            </a>
          ))}
        </div>
      )}

      {/* PRO */}
      {c.pro_affiliation !== "none" && (
        <p className="text-[10px] font-sans text-[#9b9b9b]">
          PRO: <span className="text-[#cbcbcb] uppercase">{c.pro_affiliation}</span>
          {c.pro_number && <span className="text-[#4b4b4b]"> #{c.pro_number}</span>}
        </p>
      )}

      {/* Notes */}
      {c.notes && (
        <p className="text-[10px] font-sans text-[#4b4b4b] italic">{c.notes}</p>
      )}
    </div>
  );
}
