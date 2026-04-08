const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchCandidates(phase?: string, section?: string) {
  const params = new URLSearchParams();
  if (phase) params.set("phase", phase);
  if (section) params.set("section", section);
  const res = await fetch(`${BASE}/candidates?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch candidates");
  return res.json();
}

export async function approveCandidate(id: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/approve`, { method: "POST" });
  if (!res.ok) throw new Error("Approve failed");
  return res.json();
}

export async function rejectCandidate(id: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/reject`, { method: "POST" });
  if (!res.ok) throw new Error("Reject failed");
  return res.json();
}

export async function setUseCase(id: string, use_case: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/use_case`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_case }),
  });
  if (!res.ok) throw new Error("Use case update failed");
  return res.json();
}

export async function setLabel(id: string, label: string) {
  const res = await fetch(`${BASE}/candidates/${encodeURIComponent(id)}/label`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) throw new Error("Label update failed");
  return res.json();
}

export function midiUrl(id: string) {
  return `${BASE}/midi/${encodeURIComponent(id)}`;
}
