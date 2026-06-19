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

  // For generate when no work order
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
    { id: "brief", label: "Brief" },
    { id: "collaborator", label: "Collaborator" },
    { id: "budget", label: "Budget" },
    { id: "calendar", label: "Calendar" },
  ];

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/50"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 bottom-0 z-50 w-full max-w-md bg-zinc-900 border-l border-zinc-800 flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800 flex-shrink-0">
          <h2 className="text-sm font-semibold text-white font-sans">
            {wo ? "Work Order" : "Create Work Order"}
          </h2>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 text-lg leading-none transition-colors"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800 flex-shrink-0">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2.5 text-[11px] font-sans font-medium transition-colors ${
                tab === t.id
                  ? "text-blue-300 border-b-2 border-blue-500"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {/* Generate panel — only when no work order yet */}
          {!wo && (
            <div className="flex flex-col gap-3 mb-6 p-4 rounded-lg bg-zinc-800 border border-zinc-700">
              <p className="text-[11px] font-sans text-zinc-400">
                Generate a work order pre-filled from the pipeline data.
              </p>
              <div className="flex flex-col gap-2">
                <select
                  value={selectedCollabId}
                  onChange={e => setSelectedCollabId(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
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
                    className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                  >
                    {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                  <select
                    value={selectedPlatform}
                    onChange={e => setSelectedPlatform(e.target.value as CollaboratorPlatform)}
                    className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
                  >
                    {PLATFORMS.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <button
                  onClick={handleGenerate}
                  disabled={generating || !selectedCollabId}
                  className="w-full py-1.5 text-[11px] font-sans rounded bg-blue-800 border border-blue-600 text-blue-200 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {generating ? "Generating…" : "Generate"}
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

        {/* Footer actions */}
        {wo && (
          <div className="flex items-center gap-2 px-5 py-4 border-t border-zinc-800 flex-shrink-0">
            <button
              onClick={handleSave}
              disabled={saving || !dirty}
              className="flex-1 py-2 text-[11px] font-sans font-semibold rounded bg-blue-800 border border-blue-700 text-blue-100 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {saving ? "Saving…" : dirty ? "Save" : "Saved"}
            </button>
            <button
              onClick={handleDraftEmail}
              disabled={drafting}
              className="flex-1 py-2 text-[11px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {drafting ? "Drafting…" : "Draft Email"}
            </button>
          </div>
        )}
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-[60] px-4 py-2.5 rounded-lg text-sm font-sans shadow-xl ${
            toast.type === "error"
              ? "bg-red-900 border border-red-700 text-red-200"
              : "bg-zinc-800 border border-zinc-600 text-zinc-100"
          }`}
        >
          {toast.message}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab sub-components
// ---------------------------------------------------------------------------

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[10px] font-sans text-zinc-500 uppercase tracking-wide">{label}</label>
      {children}
    </div>
  );
}

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
  const cls = "w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 resize-none";
  return multiline ? (
    <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className={cls} rows={3} />
  ) : (
    <input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className={cls} />
  );
}

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
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
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
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
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
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
        >
          {collaborators.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
      </Field>

      {collaborator && (
        <div className="p-3 rounded-lg bg-zinc-800 border border-zinc-700">
          <CollaboratorProfileCard collaborator={collaborator} />
          <Link
            href={`/collaborators`}
            className="mt-2 flex items-center gap-1 text-[10px] font-sans text-zinc-500 hover:text-blue-400 transition-colors"
          >
            <span>Edit profile</span>
            <span>→</span>
          </Link>
        </div>
      )}

      <Field label="Royalty split">
        <div className="grid grid-cols-3 gap-2">
          {(["mechanical_pct", "performance_pct", "sync_pct"] as const).map(field => (
            <div key={field} className="flex flex-col gap-1">
              <span className="text-[9px] font-sans text-zinc-600 uppercase">
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
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
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
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
          />
        </Field>
        <Field label="Paid ($)">
          <input
            type="number"
            min={0}
            value={wo.budget_paid ?? ""}
            onChange={e => onChange("budget_paid", e.target.value ? Number(e.target.value) : null)}
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
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
              className={`px-2.5 py-1 text-[10px] font-sans rounded border transition-colors capitalize ${
                wo.budget_status === s
                  ? "bg-blue-800 border-blue-600 text-blue-100"
                  : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
              }`}
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
        <p className="text-[11px] font-sans font-semibold text-zinc-400">Follow-up reminder</p>
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
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
          />
        </Field>
        <p className="text-[10px] font-sans text-zinc-600">
          Save the work order after setting a date to enable Set Reminder via GCal.
        </p>
      </div>

      <div className="border-t border-zinc-800 pt-4 flex flex-col gap-3">
        <p className="text-[11px] font-sans font-semibold text-zinc-400">Deadline</p>
        <Field label="Deadline date">
          <input
            type="date"
            value={wo.deadline ?? ""}
            onChange={e => onChange("deadline", e.target.value || null)}
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs font-sans text-zinc-200 focus:outline-none focus:border-zinc-500"
          />
        </Field>
      </div>

      {wo.calendar_event_id && (
        <p className="text-[10px] font-sans text-zinc-500">
          GCal event: <span className="text-zinc-400 font-mono">{wo.calendar_event_id}</span>
        </p>
      )}
    </div>
  );
}
