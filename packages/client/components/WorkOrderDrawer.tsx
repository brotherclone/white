"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Collaborator,
  CollaboratorPlatform,
  CollaboratorRole,
  WorkOrder,
} from "@/lib/types";
import {
  draftWorkOrderEmail,
  fetchCollaborators,
  generateWorkOrder,
  updateWorkOrder,
} from "@/lib/api";
import CollaboratorProfileCard from "./CollaboratorProfileCard";

type DrawerTab = "brief" | "collaborator" | "budget" | "calendar";

const ROLES: CollaboratorRole[] = [
  "vocalist", "drummer", "guitarist", "bassist",
  "keys", "strings", "brass", "mixing", "mastering", "other",
];

const PLATFORMS: CollaboratorPlatform[] = ["direct", "airgigs", "soundbetter", "other"];

interface Toast {
  message: string;
  type: "success" | "error";
}

interface WorkOrderDrawerProps {
  workOrder: WorkOrder | null;
  collaborator: Collaborator | null;
  onClose: () => void;
  onSaved: (wo: WorkOrder) => void;
}

const ROSE_BG: React.CSSProperties = {
  backgroundImage: "url('/rose.png')",
  backgroundRepeat: "repeat",
};

function StarIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="currentColor" aria-hidden>
      <polygon points="4.5,0 5.53,3.23 9,3.23 6.24,5.23 7.27,8.47 4.5,6.47 1.73,8.47 2.76,5.23 0,3.23 3.47,3.23" />
    </svg>
  );
}

export default function WorkOrderDrawer({
  workOrder: initialWorkOrder,
  collaborator: initialCollaborator,
  onClose,
  onSaved,
}: WorkOrderDrawerProps) {
  const [tab, setTab] = useState<DrawerTab>("brief");
  const [wo, setWo] = useState<WorkOrder | null>(initialWorkOrder);
  const [collaborator, setCollaborator] = useState<Collaborator | null>(initialCollaborator);
  const [collaborators, setCollaborators] = useState<Collaborator[]>([]);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [drafting, setDrafting] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const [dirty, setDirty] = useState(false);

  const [selectedCollabId, setSelectedCollabId] = useState(initialCollaborator?.id ?? "");
  const [selectedRole, setSelectedRole] = useState<CollaboratorRole>(
    initialWorkOrder?.role ?? "vocalist"
  );
  const [selectedPlatform, setSelectedPlatform] = useState<CollaboratorPlatform>(
    initialWorkOrder?.platform ?? "direct"
  );

  useEffect(() => {
    fetchCollaborators().then(setCollaborators).catch(() => {});
  }, []);

  function showToast(message: string, type: "success" | "error" = "success") {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3500);
  }

  function updateField<K extends keyof WorkOrder>(key: K, value: WorkOrder[K]) {
    if (!wo) return;
    setWo({ ...wo, [key]: value });
    setDirty(true);
  }

  async function handleGenerate() {
    if (!selectedCollabId) return;
    setGenerating(true);
    try {
      const generated = await generateWorkOrder(selectedCollabId, selectedRole, selectedPlatform);
      setWo(generated);
      const found = collaborators.find(c => c.id === selectedCollabId) ?? null;
      setCollaborator(found);
      setDirty(true);
    } catch (err) {
      showToast(String(err), "error");
    } finally {
      setGenerating(false);
    }
  }

  async function handleSave() {
    if (!wo) return;
    setSaving(true);
    try {
      const saved = await updateWorkOrder(wo);
      onSaved(saved);
      setDirty(false);
      showToast("Work order saved");
    } catch (err) {
      showToast(String(err), "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleDraftEmail() {
    if (!wo) return;
    setDrafting(true);
    try {
      await draftWorkOrderEmail(wo.collaborator_id);
      showToast(`Gmail draft created for ${collaborator?.name ?? wo.collaborator_id}`);
    } catch (err) {
      const msg = String(err);
      if (msg.toLowerCase().includes("no email")) {
        showToast("Add an email address to this collaborator first", "error");
      } else {
        showToast(msg, "error");
      }
    } finally {
      setDrafting(false);
    }
  }

  const tabs: { id: DrawerTab; label: string }[] = [
    { id: "brief", label: "brief" },
    { id: "collaborator", label: "collaborator" },
    { id: "budget", label: "budget" },
    { id: "calendar", label: "calendar" },
  ];

  return (
    <>
      {/* Backdrop — brand black, no bluish tint */}
      <div className="fixed inset-0 z-40 bg-black/70" onClick={onClose} />

      {/* Drawer */}
      <div
        className="fixed right-0 top-0 bottom-0 z-50 w-full max-w-md bg-[#000] border-l border-[#222] flex flex-col"
        style={{ boxShadow: "-6px 0 24px rgba(0,0,0,0.6)" }}
      >
        {/* Header — rose.png chrome surface */}
        <div
          className="flex items-center justify-between px-5 py-4 border-b border-[#222] flex-shrink-0 bg-[#000]"
          style={ROSE_BG}
        >
          <h2 className="text-sm font-display font-bold text-[#f6f6f6] flex items-center gap-2">
            <StarIcon />
            {wo ? "work order" : "create work order"}
          </h2>
          <button
            onClick={onClose}
            className="text-[#cbcbcb] hover:text-[#EF7143] text-lg leading-none transition-colors"
            style={{ transitionDuration: "300ms", transitionTimingFunction: "ease-in-out" }}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Tab bar — rose.png chrome surface */}
        <div
          className="flex border-b border-[#222] flex-shrink-0 bg-[#000]"
          style={ROSE_BG}
        >
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className="px-4 py-2.5 text-[11px] font-sans font-medium"
              style={{
                color: tab === t.id ? "#f6f6f6" : "#9b9b9b",
                boxShadow: tab === t.id ? "inset 0 -2px 0 #f6f6f6" : "none",
                transition: "color 300ms ease-in-out",
              }}
              onMouseEnter={e => {
                if (tab !== t.id) (e.currentTarget as HTMLElement).style.color = "#abd96d";
              }}
              onMouseLeave={e => {
                if (tab !== t.id) (e.currentTarget as HTMLElement).style.color = "#9b9b9b";
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 bg-[#000]">
          {/* Generate panel — only when no work order yet */}
          {!wo && (
            <div
              className="flex flex-col gap-3 mb-6 p-4 bg-[#0a0a0a] border border-[#222]"
              style={{ boxShadow: "6px 5px 9px rgba(0,0,0,0.25)" }}
            >
              <p className="text-[0.8rem] font-sans font-light text-[#cbcbcb]">
                Generate a work order pre-filled from the pipeline data.
              </p>
              <div className="flex flex-col gap-2">
                <select
                  value={selectedCollabId}
                  onChange={e => setSelectedCollabId(e.target.value)}
                  className="w-full border border-[#333] bg-[#000] px-2 py-1.5 text-[0.85rem] text-[#f6f6f6] outline-none appearance-none focus:border-[#EF7143] transition-colors"
                  style={{ fontFamily: "semplicitapro, sans-serif" }}
                >
                  <option value="">Select collaborator…</option>
                  {collaborators.map(c => (
                    <option key={c.id} value={c.id}>{c.name}</option>
                  ))}
                </select>
                <div className="flex gap-2">
                  <select
                    value={selectedRole}
                    onChange={e => setSelectedRole(e.target.value as CollaboratorRole)}
                    className="flex-1 border border-[#333] bg-[#000] px-2 py-1.5 text-[0.85rem] text-[#f6f6f6] outline-none appearance-none focus:border-[#EF7143] transition-colors"
                    style={{ fontFamily: "semplicitapro, sans-serif" }}
                  >
                    {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                  <select
                    value={selectedPlatform}
                    onChange={e => setSelectedPlatform(e.target.value as CollaboratorPlatform)}
                    className="flex-1 border border-[#333] bg-[#000] px-2 py-1.5 text-[0.85rem] text-[#f6f6f6] outline-none appearance-none focus:border-[#EF7143] transition-colors"
                    style={{ fontFamily: "semplicitapro, sans-serif" }}
                  >
                    {PLATFORMS.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <button
                  onClick={handleGenerate}
                  disabled={generating || !selectedCollabId}
                  className="w-full py-1.5 text-[11px] font-sans font-bold bg-[#abd96d] text-[#383838] hover:bg-[#9ecf5a] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  {generating ? "generating…" : "generate"}
                </button>
              </div>
            </div>
          )}

          {wo && tab === "brief" && <BriefTab wo={wo} onChange={updateField} />}
          {wo && tab === "collaborator" && (
            <CollaboratorTab
              wo={wo}
              collaborator={collaborator}
              collaborators={collaborators}
              onChange={updateField}
              onCollaboratorChange={setCollaborator}
            />
          )}
          {wo && tab === "budget" && <BudgetTab wo={wo} onChange={updateField} />}
          {wo && tab === "calendar" && <CalendarTab wo={wo} onChange={updateField} />}
        </div>

        {/* Footer — rose.png chrome surface */}
        {wo && (
          <div
            className="flex items-center gap-2 px-5 py-4 border-t border-[#222] flex-shrink-0 bg-[#000]"
            style={ROSE_BG}
          >
            <button
              onClick={handleSave}
              disabled={saving || !dirty}
              className="flex-1 py-2 text-[11px] font-sans font-bold bg-[#abd96d] text-[#383838] hover:bg-[#9ecf5a] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              {saving ? "saving…" : dirty ? "save" : "saved"}
            </button>
            <button
              onClick={handleDraftEmail}
              disabled={drafting}
              className="flex-1 py-2 text-[11px] font-sans bg-[#000] border border-[#333] text-[#cbcbcb] hover:text-[#f6f6f6] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              {drafting ? "drafting…" : "draft email"}
            </button>
          </div>
        )}
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-[60] px-4 py-2.5 text-sm font-sans shadow-xl ${
            toast.type === "error"
              ? "bg-[#AE0A33] text-[#FCFCFC]"
              : "bg-[#000] border border-[#333] text-[#f6f6f6]"
          }`}
          style={{ boxShadow: "6px 5px 9px rgba(0,0,0,0.35)" }}
        >
          {toast.message}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[10px] font-sans text-[#9b9b9b] uppercase tracking-wide">{label}</label>
      {children}
    </div>
  );
}

const inputCls =
  "w-full bg-[#000] border border-[#333] px-2.5 py-1.5 text-xs font-sans text-[#f6f6f6] placeholder-[#4b4b4b] focus:outline-none focus:border-[#EF7143] transition-colors resize-none";

function TextInput({
  value,
  onChange,
  placeholder,
  multiline,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  multiline?: boolean;
}) {
  return multiline ? (
    <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className={inputCls} rows={3} />
  ) : (
    <input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className={inputCls} />
  );
}

// ---------------------------------------------------------------------------
// Tab sub-components
// ---------------------------------------------------------------------------

function BriefTab({ wo, onChange }: { wo: WorkOrder; onChange: <K extends keyof WorkOrder>(k: K, v: WorkOrder[K]) => void }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-3 gap-3">
        <Field label="Key">
          <TextInput value={wo.key} onChange={v => onChange("key", v)} placeholder="D minor" />
        </Field>
        <Field label="BPM">
          <input
            type="number"
            value={wo.bpm ?? ""}
            onChange={e => onChange("bpm", e.target.value ? Number(e.target.value) : null)}
            className={inputCls}
          />
        </Field>
        <Field label="Time sig">
          <TextInput value={wo.time_signature} onChange={v => onChange("time_signature", v)} placeholder="4/4" />
        </Field>
      </div>
      <Field label="Sections">
        <TextInput
          value={wo.sections.join("\n")}
          onChange={v => onChange("sections", v.split("\n").filter(Boolean))}
          multiline
          placeholder="verse (8 bars)&#10;chorus (8 bars)"
        />
      </Field>
      <Field label="Creative direction">
        <TextInput value={wo.creative_direction} onChange={v => onChange("creative_direction", v)} multiline />
      </Field>
      <Field label="Part notes">
        <TextInput value={wo.part_notes} onChange={v => onChange("part_notes", v)} multiline placeholder="Role-specific notes…" />
      </Field>
      <Field label="Deliverable format">
        <TextInput value={wo.deliverable_format} onChange={v => onChange("deliverable_format", v)} placeholder="48kHz/24bit WAV, dry" />
      </Field>
      <Field label="Deadline">
        <input
          type="date"
          value={wo.deadline ?? ""}
          onChange={e => onChange("deadline", e.target.value || null)}
          className={inputCls}
        />
      </Field>
    </div>
  );
}

function CollaboratorTab({
  wo,
  collaborator,
  collaborators,
  onChange,
  onCollaboratorChange,
}: {
  wo: WorkOrder;
  collaborator: Collaborator | null;
  collaborators: Collaborator[];
  onChange: <K extends keyof WorkOrder>(k: K, v: WorkOrder[K]) => void;
  onCollaboratorChange: (c: Collaborator | null) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <Field label="Collaborator">
        <select
          value={wo.collaborator_id}
          onChange={e => {
            onChange("collaborator_id", e.target.value);
            const found = collaborators.find(c => c.id === e.target.value) ?? null;
            onCollaboratorChange(found);
          }}
          className="w-full border border-[#333] bg-[#000] px-2.5 py-1.5 text-xs text-[#f6f6f6] outline-none appearance-none focus:border-[#EF7143] transition-colors"
          style={{ fontFamily: "semplicitapro, sans-serif" }}
        >
          {collaborators.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
      </Field>

      {collaborator && (
        <div
          className="p-3 bg-[#000] border border-[#222]"
          style={{ boxShadow: "6px 5px 9px rgba(0,0,0,0.25)" }}
        >
          <CollaboratorProfileCard collaborator={collaborator} />
          <Link
            href="/collaborators"
            className="mt-2 flex items-center gap-1 text-[10px] font-sans text-[#9b9b9b] hover:text-[#EF7143] transition-colors"
            style={{ transitionDuration: "300ms" }}
          >
            <span>edit profile</span>
            <span>→</span>
          </Link>
        </div>
      )}

      <Field label="Royalty split">
        <div className="grid grid-cols-3 gap-2">
          {(["mechanical_pct", "performance_pct", "sync_pct"] as const).map(field => (
            <div key={field} className="flex flex-col gap-1">
              <span className="text-[9px] font-sans text-[#9b9b9b] uppercase">
                {field.replace("_pct", "").replace("_", " ")}
              </span>
              <input
                type="number"
                min={0}
                max={100}
                step={0.5}
                value={wo.royalty_split?.[field] ?? 0}
                onChange={e => {
                  const split = wo.royalty_split ?? {
                    collaborator_id: wo.collaborator_id,
                    song_slug: wo.song_slug,
                    mechanical_pct: 0,
                    performance_pct: 0,
                    sync_pct: 0,
                    notes: "",
                  };
                  onChange("royalty_split", { ...split, [field]: Number(e.target.value) });
                }}
                className={inputCls}
              />
            </div>
          ))}
        </div>
      </Field>
    </div>
  );
}

function BudgetTab({ wo, onChange }: { wo: WorkOrder; onChange: <K extends keyof WorkOrder>(k: K, v: WorkOrder[K]) => void }) {
  const statuses = ["pending", "agreed", "invoiced", "paid"] as const;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        <Field label="Agreed ($)">
          <input
            type="number"
            min={0}
            value={wo.budget_agreed ?? ""}
            onChange={e => onChange("budget_agreed", e.target.value ? Number(e.target.value) : null)}
            className={inputCls}
          />
        </Field>
        <Field label="Paid ($)">
          <input
            type="number"
            min={0}
            value={wo.budget_paid ?? ""}
            onChange={e => onChange("budget_paid", e.target.value ? Number(e.target.value) : null)}
            className={inputCls}
          />
        </Field>
      </div>
      <Field label="Currency">
        <TextInput value={wo.budget_currency} onChange={v => onChange("budget_currency", v)} placeholder="USD" />
      </Field>
      <Field label="Status">
        <div className="flex gap-1.5 flex-wrap">
          {statuses.map(s => (
            <button
              key={s}
              onClick={() => onChange("budget_status", s)}
              className="px-2.5 py-1 text-[10px] font-sans border transition-colors"
              style={
                wo.budget_status === s
                  ? { background: "#abd96d", borderColor: "#abd96d", color: "#383838" }
                  : { background: "#000", borderColor: "#333", color: "#9b9b9b" }
              }
            >
              {s}
            </button>
          ))}
        </div>
      </Field>
    </div>
  );
}

function CalendarTab({ wo, onChange }: { wo: WorkOrder; onChange: <K extends keyof WorkOrder>(k: K, v: WorkOrder[K]) => void }) {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-3">
        <p className="text-[11px] font-sans font-semibold text-[#9b9b9b]">follow-up reminder</p>
        <Field label="Reason">
          <TextInput
            value={wo.follow_up_reason}
            onChange={v => onChange("follow_up_reason", v)}
            placeholder="on tour until Aug 3, budget available on payday…"
          />
        </Field>
        <Field label="Follow-up date">
          <input
            type="date"
            value={wo.follow_up_date ?? ""}
            onChange={e => onChange("follow_up_date", e.target.value || null)}
            className={inputCls}
          />
        </Field>
        <p className="text-[10px] font-sans text-[#4b4b4b]">
          Save the work order after setting a date to enable Set Reminder via GCal.
        </p>
      </div>

      <div className="border-t border-[#222] pt-4 flex flex-col gap-3">
        <p className="text-[11px] font-sans font-semibold text-[#9b9b9b]">deadline</p>
        <Field label="Deadline date">
          <input
            type="date"
            value={wo.deadline ?? ""}
            onChange={e => onChange("deadline", e.target.value || null)}
            className={inputCls}
          />
        </Field>
      </div>

      {wo.calendar_event_id && (
        <p className="text-[10px] font-sans text-[#4b4b4b]">
          GCal event: <span className="text-[#9b9b9b]">{wo.calendar_event_id}</span>
        </p>
      )}
    </div>
  );
}
