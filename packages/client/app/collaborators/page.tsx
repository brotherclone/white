"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  createCollaborator, deleteCollaborator, fetchCollaborators, saveCollaborator,
} from "@/lib/api";
import {
  AvailabilityWindow, Collaborator, CollaboratorPlatform, CollaboratorRole,
  PROAffiliation, PlatformProfile,
} from "@/lib/types";

const ALL_ROLES: CollaboratorRole[] = [
  "vocalist", "drummer", "guitarist", "bassist",
  "keys", "strings", "brass", "mixing", "mastering", "other",
];
const ALL_PLATFORMS: CollaboratorPlatform[] = ["direct", "airgigs", "soundbetter", "other"];
const ALL_PRO: PROAffiliation[] = ["none", "ascap", "bmi", "sesac", "socan", "prs", "other"];

function emptyCollaborator(): Collaborator {
  return {
    id: "", name: "", roles: [], email: null, photo_url: null,
    platforms: [], website: null, socials: {}, pro_affiliation: "none",
    pro_number: null, availability_windows: [], notes: "",
  };
}

function isCurrentlyUnavailable(c: Collaborator): boolean {
  const today = new Date().toISOString().slice(0, 10);
  return c.availability_windows.some(
    w => w.unavailable_from <= today && today <= w.unavailable_until
  );
}

function cls(...parts: (string | false | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[10px] font-sans text-zinc-500 uppercase tracking-wide">{label}</label>
      {children}
    </div>
  );
}

const inputCls = "w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500";

export default function CollaboratorsPage() {
  const [collaborators, setCollaborators] = useState<Collaborator[]>([]);
  const [selected, setSelected] = useState<Collaborator | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [draft, setDraft] = useState<Collaborator>(emptyCollaborator());
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const load = useCallback(async () => {
    const list = await fetchCollaborators().catch(() => []);
    setCollaborators(list);
  }, []);

  useEffect(() => { load(); }, [load]);

  function showToast(msg: string) {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  }

  function openNew() {
    setIsNew(true);
    setSelected(null);
    setDraft(emptyCollaborator());
    setError(null);
    setConfirmDelete(false);
  }

  function openEdit(c: Collaborator) {
    setIsNew(false);
    setSelected(c);
    setDraft({ ...c });
    setError(null);
    setConfirmDelete(false);
  }

  function set<K extends keyof Collaborator>(key: K, value: Collaborator[K]) {
    setDraft(d => ({ ...d, [key]: value }));
  }

  async function handleSave() {
    setError(null);
    if (!draft.id.trim()) { setError("ID is required"); return; }
    if (!draft.name.trim()) { setError("Name is required"); return; }
    if (draft.roles.length === 0) { setError("Select at least one role"); return; }
    setSaving(true);
    try {
      if (isNew) {
        await createCollaborator(draft);
        showToast(`${draft.name} created`);
      } else {
        await saveCollaborator(draft);
        showToast(`${draft.name} saved`);
      }
      await load();
      const updated = collaborators.find(c => c.id === draft.id) ?? draft;
      setSelected({ ...draft });
      setIsNew(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!selected) return;
    setDeleting(true);
    setError(null);
    try {
      await deleteCollaborator(selected.id);
      showToast(`${selected.name} deleted`);
      setSelected(null);
      setIsNew(false);
      setConfirmDelete(false);
      await load();
    } catch (e) {
      setError(String(e));
      setConfirmDelete(false);
    } finally {
      setDeleting(false);
    }
  }

  const showForm = isNew || selected !== null;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 font-mono flex flex-col">
      {/* Header */}
      <div className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4 flex-shrink-0">
        <Link href="/" className="text-zinc-500 hover:text-zinc-300 text-xs font-sans transition-colors">
          ← home
        </Link>
        <h1 className="text-lg font-bold text-white tracking-tight">Collaborator Manager</h1>
        <button
          onClick={openNew}
          className="ml-auto px-3 py-1.5 text-xs font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-zinc-600 transition-colors"
        >
          + New Collaborator
        </button>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* Sidebar list */}
        <div className="w-64 flex-shrink-0 border-r border-zinc-800 overflow-y-auto">
          {collaborators.length === 0 ? (
            <div className="px-4 py-8 text-center text-zinc-600 text-xs font-sans">
              No collaborators yet.<br />Click &ldquo;+ New Collaborator&rdquo; to add one.
            </div>
          ) : (
            collaborators.map(c => {
              const unavailable = isCurrentlyUnavailable(c);
              const isSelected = selected?.id === c.id && !isNew;
              return (
                <button
                  key={c.id}
                  onClick={() => openEdit(c)}
                  className={cls(
                    "w-full text-left px-4 py-3 border-b border-zinc-800/60 transition-colors",
                    isSelected
                      ? "bg-zinc-800 border-l-2 border-l-blue-500"
                      : "hover:bg-zinc-900"
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    {c.photo_url ? (
                      <img src={c.photo_url} alt={c.name} className="w-7 h-7 rounded-full object-cover flex-shrink-0" />
                    ) : (
                      <div className="w-7 h-7 rounded-full bg-zinc-700 flex items-center justify-center flex-shrink-0">
                        <span className="text-[10px] font-sans font-semibold text-zinc-400">
                          {c.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase()}
                        </span>
                      </div>
                    )}
                    <span className="text-xs font-sans font-medium text-zinc-200 truncate">{c.name}</span>
                    {unavailable && (
                      <span className="ml-auto w-2 h-2 rounded-full bg-yellow-500 flex-shrink-0" title="Currently unavailable" />
                    )}
                  </div>
                  <div className="flex gap-1 flex-wrap pl-9">
                    {c.roles.slice(0, 3).map(r => (
                      <span key={r} className="text-[9px] font-sans px-1 py-0.5 rounded bg-zinc-700 text-zinc-400">{r}</span>
                    ))}
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Form panel */}
        {showForm ? (
          <div className="flex-1 overflow-y-auto px-6 py-5">
            <div className="max-w-xl">
              <h2 className="text-sm font-semibold text-white font-sans mb-4">
                {isNew ? "New Collaborator" : `Edit — ${selected?.name}`}
              </h2>

              {error && (
                <div className="mb-4 bg-red-900/40 border border-red-700 rounded p-2.5 text-xs font-sans text-red-300">
                  {error}
                </div>
              )}

              <div className="flex flex-col gap-5">
                {/* Identity */}
                <div className="grid grid-cols-2 gap-3">
                  <Field label="ID (slug, immutable)">
                    <input
                      type="text"
                      value={draft.id}
                      onChange={e => set("id", e.target.value.toLowerCase().replace(/\s+/g, "-"))}
                      placeholder="kate-koherence"
                      disabled={!isNew}
                      className={cls(inputCls, !isNew && "opacity-40 cursor-not-allowed")}
                    />
                  </Field>
                  <Field label="Display name">
                    <input type="text" value={draft.name} onChange={e => set("name", e.target.value)} placeholder="Kate Koherence" className={inputCls} />
                  </Field>
                </div>

                {/* Roles */}
                <Field label="Roles">
                  <div className="flex flex-wrap gap-1.5">
                    {ALL_ROLES.map(r => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => {
                          const roles = draft.roles.includes(r)
                            ? draft.roles.filter(x => x !== r)
                            : [...draft.roles, r];
                          set("roles", roles);
                        }}
                        className={cls(
                          "px-2.5 py-1 text-[10px] font-sans rounded border capitalize transition-colors",
                          draft.roles.includes(r)
                            ? "bg-blue-800 border-blue-600 text-blue-100"
                            : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
                        )}
                      >
                        {r}
                      </button>
                    ))}
                  </div>
                </Field>

                {/* Contact */}
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Email">
                    <input type="email" value={draft.email ?? ""} onChange={e => set("email", e.target.value || null)} placeholder="kate@example.com" className={inputCls} />
                  </Field>
                  <Field label="Website">
                    <input type="url" value={draft.website ?? ""} onChange={e => set("website", e.target.value || null)} placeholder="https://…" className={inputCls} />
                  </Field>
                </div>

                <Field label="Photo URL">
                  <input type="url" value={draft.photo_url ?? ""} onChange={e => set("photo_url", e.target.value || null)} placeholder="https://…" className={inputCls} />
                </Field>

                {/* PRO */}
                <div className="grid grid-cols-2 gap-3">
                  <Field label="PRO affiliation">
                    <select
                      value={draft.pro_affiliation}
                      onChange={e => set("pro_affiliation", e.target.value as PROAffiliation)}
                      className={inputCls}
                    >
                      {ALL_PRO.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
                    </select>
                  </Field>
                  <Field label="PRO number">
                    <input type="text" value={draft.pro_number ?? ""} onChange={e => set("pro_number", e.target.value || null)} placeholder="IPI / CAE" className={inputCls} />
                  </Field>
                </div>

                {/* Platforms */}
                <Field label="Platforms">
                  <div className="flex flex-col gap-2">
                    {draft.platforms.map((p, i) => (
                      <div key={i} className="flex gap-2 items-center">
                        <select
                          value={p.platform}
                          onChange={e => {
                            const ps = [...draft.platforms];
                            ps[i] = { ...ps[i], platform: e.target.value as CollaboratorPlatform };
                            set("platforms", ps);
                          }}
                          className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                        >
                          {ALL_PLATFORMS.map(pl => <option key={pl} value={pl}>{pl}</option>)}
                        </select>
                        <input
                          type="url"
                          value={p.url}
                          onChange={e => {
                            const ps = [...draft.platforms];
                            ps[i] = { ...ps[i], url: e.target.value };
                            set("platforms", ps);
                          }}
                          placeholder="https://…"
                          className={cls(inputCls, "flex-1")}
                        />
                        <input
                          type="text"
                          value={p.username ?? ""}
                          onChange={e => {
                            const ps = [...draft.platforms];
                            ps[i] = { ...ps[i], username: e.target.value || null };
                            set("platforms", ps);
                          }}
                          placeholder="@username"
                          className="w-28 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                        />
                        <button
                          type="button"
                          onClick={() => set("platforms", draft.platforms.filter((_, j) => j !== i))}
                          className="text-zinc-600 hover:text-red-400 text-sm leading-none flex-shrink-0 transition-colors"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => set("platforms", [...draft.platforms, { platform: "direct", url: "", username: null }])}
                      className="text-[10px] font-sans text-zinc-600 hover:text-zinc-400 transition-colors self-start"
                    >
                      + Add platform
                    </button>
                  </div>
                </Field>

                {/* Socials */}
                <Field label="Socials">
                  <div className="flex flex-col gap-2">
                    {Object.entries(draft.socials).map(([platform, url], i) => (
                      <div key={i} className="flex gap-2 items-center">
                        <input
                          type="text"
                          value={platform}
                          onChange={e => {
                            const entries = Object.entries(draft.socials);
                            entries[i] = [e.target.value, url];
                            set("socials", Object.fromEntries(entries));
                          }}
                          placeholder="instagram"
                          className="w-28 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                        />
                        <input
                          type="url"
                          value={url}
                          onChange={e => {
                            const s = { ...draft.socials };
                            s[platform] = e.target.value;
                            set("socials", s);
                          }}
                          placeholder="https://…"
                          className={cls(inputCls, "flex-1")}
                        />
                        <button
                          type="button"
                          onClick={() => {
                            const s = { ...draft.socials };
                            delete s[platform];
                            set("socials", s);
                          }}
                          className="text-zinc-600 hover:text-red-400 text-sm leading-none flex-shrink-0 transition-colors"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => set("socials", { ...draft.socials, "": "" })}
                      className="text-[10px] font-sans text-zinc-600 hover:text-zinc-400 transition-colors self-start"
                    >
                      + Add social
                    </button>
                  </div>
                </Field>

                {/* Availability windows */}
                <Field label="Availability windows">
                  <div className="flex flex-col gap-2">
                    {draft.availability_windows.map((w, i) => (
                      <div key={i} className="flex gap-2 items-center flex-wrap">
                        <input
                          type="date"
                          value={w.unavailable_from}
                          onChange={e => {
                            const ws = [...draft.availability_windows];
                            ws[i] = { ...ws[i], unavailable_from: e.target.value };
                            set("availability_windows", ws);
                          }}
                          className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                        />
                        <span className="text-zinc-600 text-xs">→</span>
                        <input
                          type="date"
                          value={w.unavailable_until}
                          onChange={e => {
                            const ws = [...draft.availability_windows];
                            ws[i] = { ...ws[i], unavailable_until: e.target.value };
                            set("availability_windows", ws);
                          }}
                          className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                        />
                        <input
                          type="text"
                          value={w.reason ?? ""}
                          onChange={e => {
                            const ws = [...draft.availability_windows];
                            ws[i] = { ...ws[i], reason: e.target.value || null };
                            set("availability_windows", ws);
                          }}
                          placeholder="reason (optional)"
                          className="flex-1 min-w-32 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                        />
                        <button
                          type="button"
                          onClick={() => set("availability_windows", draft.availability_windows.filter((_, j) => j !== i))}
                          className="text-zinc-600 hover:text-red-400 text-sm leading-none flex-shrink-0 transition-colors"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => set("availability_windows", [
                        ...draft.availability_windows,
                        { unavailable_from: "", unavailable_until: "", reason: null },
                      ])}
                      className="text-[10px] font-sans text-zinc-600 hover:text-zinc-400 transition-colors self-start"
                    >
                      + Add window
                    </button>
                  </div>
                </Field>

                {/* Notes */}
                <Field label="Notes">
                  <textarea
                    value={draft.notes}
                    onChange={e => set("notes", e.target.value)}
                    rows={3}
                    placeholder="Preferences, contract details, anything…"
                    className={cls(inputCls, "resize-none")}
                  />
                </Field>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 mt-6 pt-4 border-t border-zinc-800">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex-1 py-2 text-xs font-sans font-semibold rounded bg-blue-800 border border-blue-700 text-blue-100 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {saving ? "Saving…" : isNew ? "Create" : "Save"}
                </button>
                {!isNew && selected && (
                  confirmDelete ? (
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-sans text-zinc-400">Delete?</span>
                      <button
                        onClick={handleDelete}
                        disabled={deleting}
                        className="px-3 py-2 text-[10px] font-sans rounded bg-red-900 border border-red-700 text-red-200 hover:bg-red-800 disabled:opacity-40 transition-colors"
                      >
                        {deleting ? "…" : "Yes, delete"}
                      </button>
                      <button
                        onClick={() => setConfirmDelete(false)}
                        className="px-3 py-2 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmDelete(true)}
                      className="px-3 py-2 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-500 hover:bg-red-900/40 hover:border-red-800 hover:text-red-300 transition-colors"
                    >
                      Delete
                    </button>
                  )
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm font-sans">
            Select a collaborator or create a new one
          </div>
        )}
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 px-4 py-2.5 rounded-lg text-sm font-sans shadow-xl bg-zinc-800 border border-zinc-600 text-zinc-100">
          {toast}
        </div>
      )}
    </div>
  );
}
